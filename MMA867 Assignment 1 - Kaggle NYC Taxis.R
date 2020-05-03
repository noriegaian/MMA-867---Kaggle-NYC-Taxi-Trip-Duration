#Purpose: Kaggle NYC Taxi Challenge - Predict trip duration time as accurately as possible

library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(lubridate)
library(sqldf)
library(readxl)
library(car)
library(estimatr)
library(caret)
library(janitor)
library(glmnet)
library(geosphere)
library(esquisse)
library(MLmetrics)
library(gridExtra)

#import datasets
taxis1 <- read.csv("Queen's MMA\\MMA 867\\Assignment 1\\Kaggle - NYC Taxi Trip Duration\\train.csv", header=TRUE, sep = ",")
taxis2 <- read.csv("Queen's MMA\\MMA 867\\Assignment 1\\Kaggle - NYC Taxi Trip Duration\\test.csv", header=TRUE, sep = ",")

#investigate completed dataset
str(taxis1)
summary(taxis1)

#change necessary data types
taxis1$vendor_id <- as.factor(taxis1$vendor_id)
taxis2$vendor_id <- as.factor(taxis2$vendor_id)
taxis1$store_and_fwd_flag <- as.factor(taxis1$store_and_fwd_flag)
taxis2$store_and_fwd_flag <- as.factor(taxis2$store_and_fwd_flag)

#feature engineering
  #date (month, day of week)
taxis1$pickup_datetime <- ymd_hms(taxis1$pickup_datetime)
taxis1$dropoff_datetime <- ymd_hms(taxis1$dropoff_datetime)
taxis2$pickup_datetime <- ymd_hms(taxis2$pickup_datetime)

taxis1 <- taxis1 %>%
  mutate(pickup_month = format(pickup_datetime,"%B")) %>%
  mutate(pickup_day = format(pickup_datetime, "%A"))
taxis2 <- taxis2 %>%
  mutate(pickup_month = format(pickup_datetime,"%B")) %>%
  mutate(pickup_day = format(pickup_datetime, "%A"))

taxis1$pickup_month <- as.factor(taxis1$pickup_month)
taxis1$pickup_day <- as.factor(taxis1$pickup_day)
taxis2$pickup_month <- as.factor(taxis2$pickup_month)
taxis2$pickup_day <- as.factor(taxis2$pickup_day)

  #time (hours)
taxis1 <- taxis1 %>%
  mutate(pickup_hour = hour(pickup_datetime))
taxis2 <- taxis2 %>%
  mutate(pickup_hour = hour(pickup_datetime))

  #distance (metres) based off of longitude and latitude (using geodist package)
taxis1 <- taxis1 %>%
  mutate(distance = distHaversine(cbind(pickup_longitude, pickup_latitude), 
                                  cbind(dropoff_longitude, dropoff_latitude)))
taxis2 <- taxis2 %>%
  mutate(distance = distHaversine(cbind(pickup_longitude, pickup_latitude), cbind(dropoff_longitude, dropoff_latitude)))

#split into train and test sets
  #split randomly
sample <- sample.int(n = nrow(taxis1), size = floor(0.7*nrow(taxis1)), replace = F)
train <- taxis1[sample, ]
test <- taxis1[-sample, ]

#notice 4 outliers/extreme values in training data (exceedingly long travel duration)...subject to removal or imputation?
  #remove exceedingly long trip durations (> 1,000,000)
train <- train %>%
  filter(trip_duration < 1000000) %>%
  filter(id != "id3008062")
  #remove exceedingly long distance (> 1,000,000)
train <- train %>%
  filter(distance < 1000000) %>%
  filter(id != "id3008062")

#start with basic linear regression model
mod1 <- lm(trip_duration ~ passenger_count + store_and_fwd_flag + pickup_month + pickup_day + pickup_hour + distance, train)
summary(mod1)

#can add to test dataset as a variable
test$pred <- predict(mod1, test)

#evaluate RMSLE
RMSLE(test$pred, test$trip_duration)
  #Observation: RMSLE = 0.67...yikes

#log-linear model
mod2 <- lm(log(trip_duration) ~ passenger_count + pickup_month + pickup_day + pickup_hour + distance, train)
summary(mod2)

test$pred <- predict(mod2, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
  #Observation: RMSLE = 0.67 still

#log-log (workaround) model
mod3 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour, train)
summary(mod3)

test$pred <- predict(mod3, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
  #Observation: RMSLE = 0.54...seems better than log-sqrt model

#Attempt manual interactions
  #interaction on # passengers and distance
mod4 <- lm(log(trip_duration) ~ log(distance+1)*passenger_count + pickup_month + pickup_day + pickup_hour, train)
summary(mod4)

test$pred <- predict(mod4, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
  #Observation: RMSLE = 0.54

#slight variation with interactions
mod5 <- lm(log(trip_duration) ~ log(distance+1)*passenger_count + pickup_month*pickup_day*pickup_hour + pickup_day*pickup_hour, train)
summary(mod5)

test$pred <- predict(mod5, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
  #Observation: RMSLE = 0.54

#Attempt stepwise interactions
mod6 <- step(lm(log(trip_duration) ~ log(distance+1)*passenger_count + pickup_month*pickup_day*pickup_hour + 
                  pickup_day*pickup_hour + log(distance+1)*pickup_day, train), direction = "backward")
summary(mod6)

test$pred <- predict(mod6, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
  #Observation: Getting same RMSLE (and model) = 0.54

#Add in pickup coordinates and vendor_id
mod7 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + vendor_id, train)
summary(mod7)

test$pred <- predict(mod7, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Observation: RMSLE = 0.54...seems better than log-sqrt model

#feature engineer airports
jfk <- tibble(longitude = -73.778889, latitude = 40.639722)
laguardia <- tibble(longitude = -73.872611, latitude = 40.77725)

train <- train %>%
  mutate(jfk_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), 
                                      cbind(jfk$longitude, jfk$latitude))) %>%
  mutate(laguardia_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), 
                                            cbind(laguardia$longitude, laguardia$latitude)))
  
train <- train %>%
  mutate(jfk_trip = if_else(jfk_distance < 2000, 1, 0)) %>%
  mutate(laguardia_trip = if_else(laguardia_distance < 2000, 1, 0))

test <- test %>%
  mutate(jfk_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(jfk$longitude, jfk$latitude))) %>%
  mutate(laguardia_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(laguardia$longitude, laguardia$latitude)))

test <- test %>%
  mutate(jfk_trip = if_else(jfk_distance < 2000, 1, 0)) %>%
  mutate(laguardia_trip = if_else(laguardia_distance < 2000, 1, 0))
  
taxis2 <- taxis2 %>%
  mutate(jfk_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(jfk$longitude, jfk$latitude))) %>%
  mutate(laguardia_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(laguardia$longitude, laguardia$latitude)))

taxis2 <- taxis2 %>%
  mutate(jfk_trip = if_else(jfk_distance < 2000, 1, 0)) %>%
  mutate(laguardia_trip = if_else(laguardia_distance < 2000, 1, 0))

train$jfk_trip <- as.factor(train$jfk_trip)
train$laguardia_trip <- as.factor(train$laguardia_trip)
test$jfk_trip <- as.factor(test$jfk_trip)
test$laguardia_trip <- as.factor(test$laguardia_trip)
taxis2$jfk_trip <- as.factor(taxis2$jfk_trip)
taxis2$laguardia_trip <- as.factor(taxis2$laguardia_trip)

#feature engineer other popular gathering places (assuming less than 2km away from dropoff)
times_square <- tibble(longitude = -73.9855, latitude = 40.7580)
mad_sq_garden <- tibble(longitude = -73.9934, latitude = 40.7505)
yankee_stadium <- tibble(longitude = -73.9262, latitude = 40.8296)
empire_state <- tibble(longitude = -73.9857, latitude = 40.7484)

train <- train %>%
  mutate(times_square_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), 
                                               cbind(times_square$longitude, times_square$latitude))) %>%
  mutate(mad_sq_garden_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), 
                                                cbind(mad_sq_garden$longitude, mad_sq_garden$latitude))) %>%
  mutate(yankee_stadium_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), 
                                                 cbind(yankee_stadium$longitude, yankee_stadium$latitude))) %>%
  mutate(empire_state_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), 
                                               cbind(empire_state$longitude, empire_state$latitude)))

train <- train %>%
  mutate(times_square_trip = if_else(times_square_distance < 2000, 1, 0)) %>%
  mutate(mad_sq_garden_trip = if_else(mad_sq_garden_distance < 2000, 1, 0)) %>%
  mutate(yankee_stadium_trip = if_else(yankee_stadium_distance < 2000, 1, 0)) %>%
  mutate(empire_state_trip = if_else(empire_state_distance < 2000, 1, 0))

test <- test %>%
  mutate(times_square_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(times_square$longitude, times_square$latitude))) %>%
  mutate(mad_sq_garden_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(mad_sq_garden$longitude, mad_sq_garden$latitude))) %>%
  mutate(yankee_stadium_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(yankee_stadium$longitude, yankee_stadium$latitude))) %>%
  mutate(empire_state_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(empire_state$longitude, empire_state$latitude)))

test <- test %>%
  mutate(times_square_trip = if_else(times_square_distance < 2000, 1, 0)) %>%
  mutate(mad_sq_garden_trip = if_else(mad_sq_garden_distance < 2000, 1, 0)) %>%
  mutate(yankee_stadium_trip = if_else(yankee_stadium_distance < 2000, 1, 0)) %>%
  mutate(empire_state_trip = if_else(empire_state_distance < 2000, 1, 0))

taxis2 <- taxis2 %>%
  mutate(times_square_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(times_square$longitude, times_square$latitude))) %>%
  mutate(mad_sq_garden_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(mad_sq_garden$longitude, mad_sq_garden$latitude))) %>%
  mutate(yankee_stadium_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(yankee_stadium$longitude, yankee_stadium$latitude))) %>%
  mutate(empire_state_distance = distHaversine(cbind(dropoff_longitude, dropoff_latitude), cbind(empire_state$longitude, empire_state$latitude)))

taxis2 <- taxis2 %>%
  mutate(times_square_trip = if_else(times_square_distance < 2000, 1, 0)) %>%
  mutate(mad_sq_garden_trip = if_else(mad_sq_garden_distance < 2000, 1, 0)) %>%
  mutate(yankee_stadium_trip = if_else(yankee_stadium_distance < 2000, 1, 0)) %>%
  mutate(empire_state_trip = if_else(empire_state_distance < 2000, 1, 0))

train$times_square_trip <- as.factor(train$times_square_trip)
train$mad_sq_garden_trip <- as.factor(train$mad_sq_garden_trip)
train$yankee_stadium_trip <- as.factor(train$yankee_stadium_trip)
train$empire_state_trip <- as.factor(train$empire_state_trip)
test$times_square_trip <- as.factor(test$times_square_trip)
test$mad_sq_garden_trip <- as.factor(test$mad_sq_garden_trip)
test$yankee_stadium_trip <- as.factor(test$yankee_stadium_trip)
test$empire_state_trip <- as.factor(test$empire_state_trip)
taxis2$times_square_trip <- as.factor(taxis2$times_square_trip)
taxis2$mad_sq_garden_trip <- as.factor(taxis2$mad_sq_garden_trip)
taxis2$yankee_stadium_trip <- as.factor(taxis2$yankee_stadium_trip)
taxis2$empire_state_trip <- as.factor(taxis2$empire_state_trip)

#attempt new model with additions of airport trips
mod8 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + vendor_id + jfk_trip + laguardia_trip, train)
summary(mod8)

test$pred <- predict(mod8, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
  #Comments: STILL at 0.54

#feature engineer rush hour (4-6PM) and work hours (8AM-6PM)
train <- train %>%
  mutate(rush_hour = if_else(pickup_hour >= 16 & pickup_hour <= 18 & pickup_day %in% 
                               c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"), 1, 0)) %>%
  mutate(work_hrs = if_else(pickup_hour >=8 & pickup_hour <=18 & pickup_day %in% 
                              c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"), 1, 0))

test <- test %>%
  mutate(rush_hour = if_else(pickup_hour >= 16 & pickup_hour <= 18 & pickup_day %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"), 1, 0)) %>%
  mutate(work_hrs = if_else(pickup_hour >=8 & pickup_hour <=18 & pickup_day %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"), 1, 0))

taxis2 <- taxis2 %>%
  mutate(rush_hour = if_else(pickup_hour >= 16 & pickup_hour <= 18 & pickup_day %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"), 1, 0)) %>%
  mutate(work_hrs = if_else(pickup_hour >=8 & pickup_hour <=18 & pickup_day %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"), 1, 0))

train$rush_hour <- as.factor(train$rush_hour)
test$rush_hour <- as.factor(test$rush_hour)
taxis2$rush_hour <- as.factor(taxis2$rush_hour)
train$work_hrs <- as.factor(train$work_hrs)
test$work_hrs <- as.factor(test$work_hrs)
taxis2$work_hrs <- as.factor(taxis2$work_hrs)

#add in popular destinations and busy hours
mod9 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + vendor_id + jfk_trip + 
              laguardia_trip + rush_hour + work_hrs + times_square_trip + mad_sq_garden_trip + yankee_stadium_trip +
              empire_state_trip, train)
summary(mod9)

test$pred <- predict(mod9, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: 0.519 (another mini improvement)

#feature engineer further (add in external dataset)
  #import
fastest_routes_test <- read.csv("Queen's MMA\\MMA 867\\Assignment 1\\Kaggle - NYC Taxi Trip Duration\\external\\fastest_routes_test.csv", header=TRUE, sep = ",")
fastest_routes_train_part_1 <- read.csv("Queen's MMA\\MMA 867\\Assignment 1\\Kaggle - NYC Taxi Trip Duration\\external\\fastest_routes_train_part_1.csv", header=TRUE, sep = ",")
fastest_routes_train_part_2 <- read.csv("Queen's MMA\\MMA 867\\Assignment 1\\Kaggle - NYC Taxi Trip Duration\\external\\fastest_routes_train_part_2.csv", header=TRUE, sep = ",")

  #union training external data
fastest_routes_train <- bind_rows(fastest_routes_train_part_1, fastest_routes_train_part_2)

  #change necessary data types
fastest_routes_test$starting_street <- as.factor(fastest_routes_test$starting_street)
fastest_routes_test$end_street <- as.factor(fastest_routes_test$end_street)
fastest_routes_train$starting_street <- as.factor(fastest_routes_train$starting_street)
fastest_routes_train$end_street <- as.factor(fastest_routes_train$end_street)

  #join onto our test and train
train <- train %>%
  left_join(fastest_routes_train, by = "id")

test <- test %>%
  left_join(fastest_routes_train, by = "id")

taxis2 <- taxis2 %>%
  left_join(fastest_routes_test, by = "id")

#build new model incorporating external data
mod10 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + vendor_id + jfk_trip + 
              laguardia_trip + rush_hour + total_distance + total_travel_time + number_of_steps, train)
summary(mod10)

test$pred <- predict(mod10, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: We've got RMSE = 0.50 (pretty good decrease here!)

#feature engineer # turns and left turns
train <- train %>%
  mutate(left_turns = str_count(step_direction, "left"),
         turns = str_count(step_maneuvers, "turn"))

test <- test %>%
  mutate(left_turns = str_count(step_direction, "left"),
         turns = str_count(step_maneuvers, "turn"))

taxis2 <- taxis2 %>%
  mutate(left_turns = str_count(step_direction, "left"),
         turns = str_count(step_maneuvers, "turn"))

#build new model incorporating # turns and left turns
mod11 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + vendor_id + jfk_trip + 
              laguardia_trip + rush_hour + work_hrs + times_square_trip + mad_sq_garden_trip + yankee_stadium_trip + 
              empire_state_trip + total_distance + total_travel_time + number_of_steps + left_turns + turns, train)
summary(mod11)

test$pred <- predict(mod11, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: We've got RMSE = 0.50

#feature engineer direction
train <- train %>%
  mutate(direction = bearing(cbind(pickup_longitude, pickup_latitude), 
                             cbind(dropoff_longitude, dropoff_latitude)))
  
test <- test %>%
  mutate(direction = bearing(cbind(pickup_longitude, pickup_latitude), 
                             cbind(dropoff_longitude, dropoff_latitude)))

taxis2 <- taxis2 %>%
  mutate(direction = bearing(cbind(pickup_longitude, pickup_latitude), 
                             cbind(dropoff_longitude, dropoff_latitude)))

#build new model incorporating direction of travel
mod12 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + vendor_id + jfk_trip + 
              laguardia_trip + rush_hour + total_distance + total_travel_time + number_of_steps + left_turns + 
              turns + direction, train)
summary(mod12)

test$pred <- predict(mod12, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: We've got RMSE = 0.50 (same)

#play with more transformations
mod13 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + vendor_id + jfk_trip + 
              laguardia_trip + rush_hour + sqrt(total_distance) + log(total_travel_time+1) + number_of_steps + left_turns + 
              turns + direction, train)
summary(mod13)

test$pred <- predict(mod13, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: We've got RMSE = 0.494 (good improvement)

#play with more transformations
mod14 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + vendor_id + jfk_trip + 
              laguardia_trip + rush_hour + sqrt(total_distance) + sqrt(total_travel_time) + number_of_steps + left_turns + 
              turns + direction, train)
summary(mod14)

test$pred <- predict(mod14, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: We've got RMSE = 0.491 (some more improvement)

#add in pickup distance from airport features and store_&_flag binary and logging direction
mod15 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + vendor_id + jfk_trip + 
              laguardia_trip + rush_hour + sqrt(total_distance) + sqrt(total_travel_time) + number_of_steps + 
              left_turns + turns + log(direction+180.01) + jfk_distance + laguardia_distance + store_and_fwd_flag, train)
summary(mod15)

test$pred <- predict(mod15, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: We've got RMSE = 0.488

#add in interactions (trial and error)
mod16 <- lm(log(trip_duration) ~ log(distance+1)*passenger_count*vendor_id*number_of_steps*left_turns*turns + pickup_month + pickup_day*pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + jfk_trip*sqrt(total_travel_time) + 
              laguardia_trip*sqrt(total_travel_time) + sqrt(total_distance)*sqrt(total_travel_time)*log(direction+180.01)*rush_hour*vendor_id + 
              jfk_distance + laguardia_distance + store_and_fwd_flag, train)
summary(mod16)

test$pred <- predict(mod16, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: We've got RMSE = 0.464

#feature engineer busiest streets (selecting handful based on a quick google search)
train <- train %>%
  mutate(busy_street = if_else(grepl('East 34th Street|5th Avenue|East 42nd Street|8th Avenue', 
                                     street_for_each_step),1,0))

test <- test %>%
  mutate(busy_street = if_else(grepl('East 34th Street|5th Avenue|East 42nd Street|8th Avenue', 
                                     street_for_each_step),1,0))

taxis2 <- taxis2 %>%
  mutate(busy_street = if_else(grepl('East 34th Street|5th Avenue|East 42nd Street|8th Avenue', 
                                     street_for_each_step),1,0))

train$busy_street <- as.factor(train$busy_street)
test$busy_street <- as.factor(test$busy_street)
taxis2$busy_street <- as.factor(taxis2$busy_street)

#add in busy streets (no interactions yet)
mod17 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + vendor_id + jfk_trip + 
              laguardia_trip + rush_hour + sqrt(total_distance) + sqrt(total_travel_time) + number_of_steps + 
              left_turns + turns + log(direction+180.01) + jfk_distance + laguardia_distance + store_and_fwd_flag + 
              yankee_stadium_trip + mad_sq_garden_trip + empire_state_trip + times_square_trip + log(yankee_stadium_distance+1) + 
              log(mad_sq_garden_distance+1) + log(empire_state_distance+1) + log(times_square_distance+1) + 
              busy_street, train)
summary(mod17)

test$pred <- predict(mod17, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: We've got RMSE = 0.479

#re-add in interactions
mod18 <- lm(log(trip_duration) ~ log(distance+1)*passenger_count*vendor_id*number_of_steps*left_turns*turns + pickup_month + pickup_day*pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + jfk_trip*sqrt(total_travel_time) + 
              laguardia_trip*sqrt(total_travel_time) + sqrt(total_distance)*sqrt(total_travel_time)*log(direction+180.01)*rush_hour*vendor_id + 
              jfk_distance + laguardia_distance + store_and_fwd_flag + yankee_stadium_trip + mad_sq_garden_trip + 
              empire_state_trip + times_square_trip + log(yankee_stadium_distance+1) + log(mad_sq_garden_distance+1) + 
              log(empire_state_distance+1) + log(times_square_distance+1) + busy_street, train)
summary(mod18)

test$pred <- predict(mod18, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
  #Comments: We've now achieve RMSE = 0.455 (more nice improvement here)

#create mini modelling df
mini_sample <- sample.int(n = nrow(train), size = floor(0.10*nrow(train)), replace = F)
mini_train <- train[mini_sample, ]

#add in external 2016 nyc weather data
nyc_weather <- read.csv("Queen's MMA\\MMA 867\\Assignment 1\\Kaggle - NYC Taxi Trip Duration\\external\\nyc_weather2016.csv", header=TRUE, sep = ",")

nyc_weather$date <- as.Date(nyc_weather$date)

train <- train %>%
  mutate(date = as.Date(pickup_datetime))
test <- test %>%
  mutate(date = as.Date(pickup_datetime))
taxis2 <- taxis2 %>%
  mutate(date = as.Date(pickup_datetime))

train <- train %>%
  left_join(nyc_weather, by = "date")

test <- test %>%
  left_join(nyc_weather, by = "date")

taxis2 <- taxis2 %>%
  left_join(nyc_weather, by = "date")

#add in weather data (no interactions yet)
mod19 <- lm(log(trip_duration) ~ log(distance+1) + passenger_count + pickup_month + pickup_day + pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + vendor_id + jfk_trip + 
              laguardia_trip + rush_hour + sqrt(total_distance) + sqrt(total_travel_time) + number_of_steps + 
              left_turns + turns + log(direction+180.01) + jfk_distance + laguardia_distance + store_and_fwd_flag + 
              yankee_stadium_trip + mad_sq_garden_trip + empire_state_trip + times_square_trip + log(yankee_stadium_distance+1) + 
              log(mad_sq_garden_distance+1) + log(empire_state_distance+1) + log(times_square_distance+1) + 
              busy_street + average.temperature + precipitation + snow.fall + snow.depth, train)
summary(mod19)

test$pred <- predict(mod19, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: We've got RMSE = 0.478

#re-add in interactions
mod20 <- lm(log(trip_duration) ~ log(distance+1)*passenger_count*vendor_id*number_of_steps*left_turns*turns*work_hrs + pickup_month*pickup_day*pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + jfk_trip*sqrt(total_travel_time)*pickup_day + 
              laguardia_trip*sqrt(total_travel_time)*pickup_day + sqrt(total_distance)*sqrt(total_travel_time)*log(direction+180.01)*rush_hour*vendor_id*work_hrs + 
              jfk_distance + laguardia_distance + yankee_stadium_trip*sqrt(total_travel_time) + mad_sq_garden_trip*sqrt(total_travel_time) + 
              empire_state_trip*sqrt(total_travel_time) + times_square_trip*sqrt(total_travel_time) + log(yankee_stadium_distance+1) + log(mad_sq_garden_distance+1) + 
              log(empire_state_distance+1) + log(times_square_distance+1) + pickup_hour*busy_street + average.temperature + 
              precipitation + snow.fall*sqrt(total_travel_time)*log(distance+1) + snow.depth*sqrt(total_travel_time)*log(distance+1), mini_train)
summary(mod20)

test$pred <- predict(mod20, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
  #Comments: RMSLE = 0.4335 (!!!)

#slight tinkering (trial and error) with interactions
mod21 <- lm(log(trip_duration) ~ log(distance+1)*passenger_count*vendor_id*number_of_steps*left_turns*turns*work_hrs*sqrt(total_travel_time) + pickup_month*pickup_day*pickup_hour +
              pickup_longitude + pickup_latitude + dropoff_longitude + dropoff_latitude + jfk_trip*sqrt(total_travel_time)*pickup_day + 
              laguardia_trip*sqrt(total_travel_time)*pickup_day + sqrt(total_distance)*sqrt(total_travel_time)*log(direction+180.01)*rush_hour*vendor_id*work_hrs + 
              jfk_distance + laguardia_distance + yankee_stadium_trip*sqrt(total_travel_time) + mad_sq_garden_trip*sqrt(total_travel_time) + 
              empire_state_trip*sqrt(total_travel_time) + times_square_trip*sqrt(total_travel_time) + log(yankee_stadium_distance+1) + log(mad_sq_garden_distance+1) + 
              log(empire_state_distance+1) + log(times_square_distance+1) + log(direction+180.01)*number_of_steps*pickup_hour*busy_street + average.temperature + 
              precipitation + snow.fall*sqrt(total_travel_time)*log(distance+1) + snow.depth*sqrt(total_travel_time)*log(distance+1), mini_train)
summary(mod21)

test$pred <- predict(mod21, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
  #Comments: RMSLE = 0.4322

#more slight tinkering with interactions
mod22 <- lm(log(trip_duration) ~ log(distance+1)*passenger_count*vendor_id*number_of_steps*left_turns*turns*work_hrs*sqrt(total_travel_time) + 
              sqrt(total_distance)*sqrt(total_travel_time)*log(direction+180.01)*rush_hour*vendor_id*work_hrs + 
              log(direction+180.01)*number_of_steps*pickup_hour*busy_street + pickup_month*pickup_day*pickup_hour + 
              jfk_trip*sqrt(total_travel_time)*pickup_day + laguardia_trip*sqrt(total_travel_time)*pickup_day + 
              yankee_stadium_trip*sqrt(total_travel_time)*pickup_hour + mad_sq_garden_trip*sqrt(total_travel_time)*pickup_hour + 
              empire_state_trip*sqrt(total_travel_time)*pickup_hour + times_square_trip*sqrt(total_travel_time)*pickup_hour + 
              snow.fall*sqrt(total_travel_time)*log(distance+1) + snow.depth*sqrt(total_travel_time)*log(distance+1) + 
              pickup_longitude*vendor_id + pickup_latitude*vendor_id + dropoff_longitude*vendor_id + dropoff_latitude*vendor_id + 
              jfk_distance*sqrt(total_travel_time) + laguardia_distance*sqrt(total_travel_time) + 
              log(yankee_stadium_distance+1)*sqrt(total_travel_time) + log(mad_sq_garden_distance+1)*sqrt(total_travel_time) + 
              log(empire_state_distance+1)*sqrt(total_travel_time) + log(times_square_distance+1)*sqrt(total_travel_time) + 
              average.temperature + precipitation, mini_train)
summary(mod22)

test$pred <- predict(mod22, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)
#Comments: RMSLE = 0.4306 (winner)

#try bootstepAIC
model23 <- stepAIC(mod22, direction='both')

summary(model22)

test$pred <- predict(model28, test)

test <- test %>%
  mutate(pred = exp(pred))

RMSLE(test$pred, test$trip_duration)

#Attempt LASSO
#create the y variable and matrix (capital X) of x variables (will make the code below easier to read + will ensure that all interactions exist)
y<-log(train$trip_duration)

#create a "dumb" model with random interactions for the matrix (first with reg carat weight, then sqrt of carat weight, then log of carat weight, then random interactions between categoricals)
#regressed on ID instead of price, since we don't have price for the full data
#we don't care that the model structure doesn't make sense here as the purpose of this feature engineering is to create all hypothetically necessary variable combinations
taxis_full <- bind_rows(train, test)
taxis_full <- taxis_full %>%
  select(-pred)
taxis_full <- bind_rows(taxis_full, taxis2)

taxis_full <- taxis_full %>%
  mutate(ID_index = rownames(taxis_full))

X<-model.matrix(ID_index ~ distance*passenger_count*vendor_id*number_of_steps + sqrt(distance)*passenger_count*vendor_id*number_of_steps + log(distance+1)*passenger_count*vendor_id*number_of_steps + pickup_month*pickup_day*pickup_hour, taxis_full)[,-1]
X<-cbind(taxis_full$ID_index,X)

# split X into testing, training/holdout and prediction as before
X.training<-subset(X,X[,1]<=1021045)
X.testing<-subset(X, (X[,1]>=1021046 & X[,1]<=1458639))
X.prediction<-subset(X,X[,1]>=1458640)

##LASSO (alpha = 1)
lasso.fit <- glmnet(x = X.training, y = y, alpha = 1)
plot(lasso.fit, xvar = "lambda")

#visualization
  #histogram of trip_duration
taxis1 %>%
  filter(trip_duration >= 0L & trip_duration <= 5000L) %>%
  ggplot() +
  aes(x = trip_duration) +
  geom_histogram(bins = 30L, fill = "#0c4c8a") +
  theme_minimal() + 
  labs(x = "trip_duration (sec)")

  #trip_duration by passenger_count and vendor+_id
taxis1 %>%
  filter(trip_duration >= 0L & trip_duration <= 5000L) %>%
  ggplot() +
  aes(x = as.factor(passenger_count), y = trip_duration, colour = vendor_id) +
  geom_boxplot(size = 1L) +
  theme_minimal() +
  labs(x = "passenger_count",
       y = "trip_duration (sec)") +
  labs(color='vendor_id')

  #trip_duration by store_and_fwd_flag
taxis1 %>%
  filter(trip_duration >= 0L & trip_duration <= 5000L) %>%
  ggplot() +
  aes(x = store_and_fwd_flag, y = trip_duration, fill = store_and_fwd_flag) +
  geom_boxplot(size = 1L) +
  theme_minimal() +
  labs(x = "store_and_fwd_flag",
       y = "trip_duration (sec)")

  #trip_duration by month
taxis1$pickup_month = factor(taxis1$pickup_month, levels = c("January", "February", "March", "April", "May", "June"))

taxis1 %>%
  filter(trip_duration >= 0L & trip_duration <= 5000L) %>%
  ggplot() +
  aes(x = pickup_month, y = trip_duration, colour = pickup_month) +
  geom_boxplot(size = 1L) +
  theme_minimal() +
  labs(x = "pickup_month",
       y = "trip_duration (sec)")

  #trip_duration by day of week and hour
taxis1$pickup_day <- factor(taxis1$pickup_day, levels= c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

avg_duration_day_hour <- taxis1 %>%
  mutate(count = 1) %>%
  group_by(pickup_day, pickup_hour) %>%
  summarise(count = sum(count), trip_duration = sum(trip_duration))

avg_duration_day_hour <- avg_duration_day_hour %>%
  mutate(avg_trip_duration = trip_duration/count)

ggplot(avg_duration_day_hour) +
  aes(x = pickup_hour, y = avg_trip_duration, colour = pickup_day) +
  geom_line(size = 1L) +
  scale_color_hue() +
  theme_minimal() +
  labs(y = "avg_trip_duration (sec)")

  #trip_duration x distance
taxis1 %>%
  filter(trip_duration >= 0L & trip_duration <= 10000L & distance < 50000L) %>%
  ggplot() +
  aes(x = distance, y = trip_duration) +
  geom_point() +
  theme_minimal() +
  labs(x = "distance (m)",
       y = "trip_duration (sec)") +
  geom_smooth(method='lm')

  #trip_duration x airport vicinity
plot5 <- train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = jfk_trip, y = trip_duration, colour = jfk_trip) + 
  geom_boxplot() +
  theme_minimal() +
  labs(x = "jfk_trip",
       y = "trip_duration (sec)") +
  scale_y_log10()

plot6 <- train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = laguardia_trip, y = trip_duration, colour = laguardia_trip) + 
  geom_boxplot() +
  theme_minimal() +
  labs(x = "laguardia_trip",
       y = "trip_duration (sec)") +
  scale_y_log10()

grid.arrange(plot5, plot6, ncol=2)

  #trip duration x other major destinations
plot1 <- train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = times_square_trip, y = trip_duration, colour = times_square_trip) +
  geom_boxplot() +
  theme_minimal() +
  labs(x = "time_square_trip",
       y = "trip_duration (sec)") +
  scale_y_log10()
plot2 <- train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = mad_sq_garden_trip, y = trip_duration, colour = mad_sq_garden_trip) +
  geom_boxplot() +
  theme_minimal() +
  labs(x = "mad_sq_garden_trip",
       y = "trip_duration (sec)") +
  scale_y_log10()
plot3 <- train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = yankee_stadium_trip, y = trip_duration, colour = yankee_stadium_trip) +
  geom_boxplot() +
  theme_minimal() +
  labs(x = "yankee_stadium_trip",
       y = "trip_duration (sec)") +
  scale_y_log10()
plot4 <- train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = empire_state_trip, y = trip_duration, colour = empire_state_trip) +
  geom_boxplot() +
  theme_minimal() +
  labs(x = "empire_state_trip",
       y = "trip_duration (sec)") +
  scale_y_log10()
grid.arrange(plot1, plot2, plot3, plot4, ncol=2)

#trip_duration x airport vicinity
plot7 <- train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = rush_hour, y = trip_duration, colour = rush_hour) + 
  geom_boxplot() +
  theme_minimal() +
  labs(x = "rush_hour",
       y = "trip_duration (sec)") +
  scale_y_log10()

plot8 <- train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = work_hrs, y = trip_duration, colour = work_hrs) + 
  geom_boxplot() +
  theme_minimal() +
  labs(x = "work_hrs",
       y = "trip_duration (sec)") +
  scale_y_log10()

grid.arrange(plot7, plot8, ncol=2)

  #trip_duration x total_distance
plot9 <- train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = total_distance, y = trip_duration) +
  geom_point() +
  theme_minimal() +
  labs(x = "total_distance (m)",
       y = "trip_duration (sec)") +
  geom_smooth(method='lm') +
  scale_y_log10() +
  scale_x_log10()

plot10 <- train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = total_travel_time, y = trip_duration) + 
  geom_point() +
  theme_minimal() + 
  labs(x = "total_travel_time (sec)",
       y = "trip_duration (sec)") +
  geom_smooth(method='lm') +
  scale_y_log10() +
  scale_x_log10()

grid.arrange(plot9, plot10, ncol=2)

  #trip_duration x left_turns
train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = left_turns, y = trip_duration) +
  geom_point() +
  theme_minimal() +
  labs(x = "left_turns",
       y = "trip_duration (sec)") +
  geom_smooth(method='lm') +
  scale_y_log10()

  #trip_duration x direction
train %>%
  filter(trip_duration >= 0L) %>%
  ggplot() +
  aes(x = direction, y = trip_duration) +
  geom_point() +
  theme_minimal() +
  labs(x = "direction (degrees)",
       y = "trip_duration (sec)") +
  stat_smooth() +
  scale_y_log10()

  #trip_duration by busy_street (zoomed)
train %>%
  filter(trip_duration >= 0L & trip_duration <= 3000L) %>%
  ggplot() +
  aes(x = busy_street, y = trip_duration, colour = busy_street) + 
  geom_boxplot() +
  theme_minimal() +
  labs(x = "busy_street",
       y = "trip_duration (sec)")

  #trip_duration by temperature
daily_weather <- train %>%
  group_by(train$date) %>%
  summarise(trip_duration = mean(trip_duration), average.temperature = mean(average.temperature), 
            precipitation = min(precipitation), snow.fall = min(snow.fall), snow.depth = min(snow.depth))

plot11 <- daily_weather %>%
  ggplot() +
  aes(x = average.temperature, y = trip_duration) +
  geom_point() +
  theme_minimal() +
  labs(x = "average.temperature (Fahrenheit)",
       y = "avg trip_duration (sec)") +
  geom_smooth(method = 'lm')

  #trip_duration by precipitation
plot12 <- daily_weather %>%
  ggplot() +
  aes(x = precipitation, y = trip_duration) +
  geom_point() +
  theme_minimal() +
  labs(x = "precipitation (inches)",
       y = "avg trip_duration (sec)") +
  geom_smooth(method = 'lm')

  #trip_duration by snow.fall
plot13 <- daily_weather %>%
  ggplot() +
  aes(x = snow.fall, y = trip_duration) +
  geom_point() +
  theme_minimal() +
  labs(x = "snow.fall (inches)",
       y = "avg trip_duration (sec)") +
  geom_smooth(method = 'lm')

#trip_duration by snow.depth
plot14 <- daily_weather %>%
  ggplot() +
  aes(x = snow.depth, y = trip_duration) +
  geom_point() +
  theme_minimal() +
  labs(x = "snow.depth (inches)",
       y = "avg trip_duration (sec)") +
  geom_smooth(method = 'lm')

grid.arrange(plot11, plot12, plot13, plot14, ncol=2)

#generate and export predictions
pred2 <- predict(mod33, taxis2)

taxis2 <- taxis2 %>%
  mutate(pred = exp(pred2))

  #export prediction results
write.csv(taxis2, file = paste0("Queen's MMA\\MMA 867\\Assignment 1\\Predictions5.csv"), row.names = FALSE, na = "")