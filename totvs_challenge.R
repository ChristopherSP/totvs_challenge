library(jsonlite)
library(data.table)
library(stringi)
library(lubridate)
library(arules)
library(caret)
library(pROC)
library(PRROC)

###################
# Data Correction
###################
data = as.data.table(read_json("./challenge.json", simplifyVector = T))

data[, register_date := as.Date(substr(register_date, 1, 10), format = "%Y-%m-%d")]

setorder(data, customer_code, register_date)

summary(data)
data[, branch_id := NULL]

factor_cols = c(
  "customer_code",
  "sales_channel",
  "seller_code",
  "order_id",
  "item_code",
  "group_code",
  "segment_code",
  "is_churn"
)

data[customer_code == '0' &
       register_date == '2008-09-23'][order(item_code)]
data[customer_code == '158' &
       register_date == '2015-10-04'][order(item_code)]
data[customer_code == '837' &
       register_date == '2017-02-09'][order(item_code)]

fixed_data = copy(data)
fixed_data = fixed_data[, .(quantity = sum(quantity),
                            unit_price = mean(unit_price)), by = c(factor_cols, 'register_date')]
fixed_data[, item_total_price := quantity * unit_price]
fixed_data[, total_price := sum(item_total_price), by = .(customer_code, register_date, order_id)]
fixed_data[, monthly_date := as.Date(paste0(
  lubridate::year(register_date),
  '-',
  lubridate::month(register_date),
  '-01'
),
format = "%Y-%m-%d")]

fixed_data[customer_code == '0' &
             register_date == '2008-09-23'][order(item_code)]

###################
# Feature Engenering
###################
fixed_data[, is_churn := as.integer(is_churn)]

agg_data = fixed_data[, .(
  n_channel = uniqueN(sales_channel),
  n_seller = uniqueN(seller_code),
  n_order = uniqueN(order_id),
  n_item = uniqueN(item_code),
  group_code = mean(group_code),
  segment_code = mean(segment_code),
  is_churn = max(as.integer(is_churn)),
  total_item = sum(quantity),
  total_value = sum(total_price)
), by = .(customer_code, monthly_date)]

agg_data[, will_abandon := is_churn]
agg_data[, max_date := max(monthly_date), by = customer_code]
agg_data[will_abandon == 1 &
           monthly_date != max_date, will_abandon := 0]

agg_data[, c(factor_cols[factor_cols %in% names(agg_data)], 'will_abandon') := lapply(.SD, as.factor), .SDcols = c(factor_cols[factor_cols %in%names(agg_data)], 'will_abandon')]

tx_variables = c("n_channel", "n_seller", "n_item", "total_item", "total_value")
by_order = paste0("tx_", tx_variables, "_by_order")

churn_data = copy(agg_data)
churn_data[, c(by_order) := lapply(.SD, function(col)
  col / n_order), .SDcols = c(tx_variables)]
churn_data[, time_between_shopping := as.numeric(difftime(monthly_date, shift(monthly_date, 1), units = 'days')), by = customer_code]

first_derivative_variables = c(tx_variables, by_order)
first_derivative_names = paste0("dr1_", first_derivative_variables)
diff_names = paste0("diff_", first_derivative_variables)

churn_data[, c(diff_names) := lapply(.SD, function(col)
  col - shift(col, 1)), by = customer_code, .SDcols = c(first_derivative_variables)]

churn_data[, c(first_derivative_names) := lapply(.SD, function(col)
  (col - shift(col, 1)) / time_between_shopping), by = customer_code, .SDcols = c(first_derivative_variables)]

last_observations_tx = paste0('last_obs_tx_', tx_variables)

setorder(churn_data, customer_code, -monthly_date)
last_observations_tx_data = churn_data[, lapply(.SD, function(col)
  ifelse(
    sum(col[1:3], na.rm = T) / sum(col[4:6], na.rm = T) == Inf,
    0,
    sum(col[1:3], na.rm = T) / sum(col[4:6], na.rm = T)
  )), by = customer_code, .SDcols = c(tx_variables)]

names(last_observations_tx_data) = c('customer_code', last_observations_tx)
churn_data = merge(churn_data,
                   last_observations_tx_data,
                   all.x = T,
                   by = 'customer_code')

filtered_data = churn_data[monthly_date == max_date][!is.na(is_churn)]

filtered_data[, lapply(.SD, function(x)
  sum(is.na(x)))]
filtered_data[is.na(filtered_data)] = 0


filtered_data[, `:=` (
  is_churn = NULL,
  max_date = NULL,
  monthly_date = NULL,
  customer_code = NULL
)]

numerical_cols = c(
  tx_variables,
  'n_order',
  by_order,
  diff_names,
  'time_between_shopping',
  first_derivative_names,
  last_observations_tx
)

###################
# Pre Processing
###################
preprocess_params = caret::preProcess(filtered_data[, .SD, .SDcols = numerical_cols],
                                      method = c("center", "scale", "pca"),
                                      thresh = 0.95)

preprocess_params

transformed = predict(preprocess_params, filtered_data)

###################
# Churn Model
###################
seed = 112358
metric = "Accuracy"
control = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  search = "grid"
)
tunegrid = expand.grid(.mtry = c(1:ncol(transformed)))

idx = createDataPartition(transformed$will_abandon, p = 0.8, list = FALSE)
training = transformed[idx, ]
test = transformed[-idx, ]

set.seed(seed)
rf_gridsearch = train(
  will_abandon ~ .,
  data = training,
  method = "rf",
  metric = metric,
  tuneGrid = tunegrid,
  trControl = control
)
print(rf_gridsearch)
plot(rf_gridsearch)

###################
# Performance Evaluation
###################
output_training = copy(training)
output_test = copy(test)

test_prediction_probability = predict(rf_gridsearch, test, type = "prob")

output_training[, predicted := predict(rf_gridsearch, training)]
output_test[, predicted := predict(rf_gridsearch, test)]
output_test[, prob := pmax(test_prediction_probability$`0`,
                           test_prediction_probability$`1`)]

cm_training = confusionMatrix(output_training$predicted, output_training$will_abandon)
cm_test = confusionMatrix(output_test$predicted, output_test$will_abandon)

roc = with(output_test, roc(will_abandon, prob))
plot(roc)

pr_positive = pr.curve(
  scores.class0 = output_test[will_abandon == 0]$prob,
  scores.class1 = output_test[will_abandon == 1]$prob,
  curve = TRUE,
  max.compute = T,
  min.compute = T,
  rand.compute = T
)

plot(pr_positive)
pr_negative = pr.curve(
  scores.class0 = 1 - output_test[will_abandon == 0]$prob,
  scores.class1 = 1 - output_test[will_abandon == 1]$prob,
  curve = TRUE,
  max.compute = T,
  min.compute = T,
  rand.compute = T
)

plot(pr_negative)

varImp(rf_gridsearch)

