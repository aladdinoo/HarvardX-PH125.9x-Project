#############################################################
# HarvardX: PH125.9 - Data Science: Capstone 
##############Predicting-Mental-Health#######################
#############################################################
#
# The following script uses the Mental Health Survey Data Set
# This dataset tests different models to predict instances of depression
# in individuals based on various parameters.
# This code was run on Windows 10 OS with RStudio Build 394
#
#
#############################################################


# Install necessary packages if not already installed
packages <- c("dplyr", "tidyverse", "readxl", "tinytex", "e1071", "randomForest",
              "rsample", "xgboost", "adabag", "data.table", "caret", 
              "ggplot2", "tidyr", "stringr", "forcats", "gridExtra")

new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load necessary libraries
library(dplyr)
library(tidyverse)
library(readxl)
library(tinytex)
library(e1071)
library(randomForest)
library(rsample)
library(xgboost)
library(adabag)
library(data.table)
library(caret)
library(readr)      # For reading data
library(ggplot2)    # For plotting
library(tidyr)      # For tidying data
library(stringr)    # For string manipulation
library(forcats)     # For factor handling
library(gridExtra)  # For arranging plots

# Suppress warnings
options(warn = -1)

# Install readr package if not already installed
if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr")
}

# Load the data
train_url <- "https://github.com/aladdinoo/HarvardX-PH125.9x-Project/blob/main/train.csv"  
test_url  <- "https://github.com/aladdinoo/HarvardX-PH125.9x-Project/blob/main/test.csv"   

# Display the first 6 rows of the dataset
train_df <- read_csv(train_url)

# Get data format
data_shape <- dim(train_df)
cat("The train_df has", data_shape[1], "rows and", data_shape[2], "columns.\n")

# Use glimpse to get information about columns
glimpse(train_df)

# Optionally, you can summarize the dataset
summary(train_df)


# ==============================================================================
# 0 Initialization
# ==============================================================================

# Assuming train_df is your data frame
gender_counts <- table(train_df$Gender)

# Print the counts
print(gender_counts)

# Count the occurrences in the "Family History of Mental Illness" column
family_history_counts <- table(train_df$`Family History of Mental Illness`)

# Print the counts
print(family_history_counts)

# Count the occurrences in the "Working Professional or Student" column
working_status_counts <- table(train_df$`Working Professional or Student`)

# Print the counts
print(working_status_counts)

# Count the occurrences in the "Have you ever had suicidal thoughts?" column
suicidal_thoughts_counts <- table(train_df$`Have you ever had suicidal thoughts ?`)

# Print the counts
print(suicidal_thoughts_counts)

# Count the occurrences in the "Dietary Habits" column
dietary_habits_counts <- table(train_df$`Dietary Habits`)

# Print the counts
print(dietary_habits_counts)


# Define valid dietary habits
valid_dietary_habits <- c("Healthy", "Unhealthy", "Moderate")

# Replace invalid entries with NA
train_df$`Dietary Habits` <- ifelse(train_df$`Dietary Habits` %in% valid_dietary_habits, 
                                     train_df$`Dietary Habits`, 
                                     NA)

# Print the counts
print(table(train_df$`Dietary Habits`))


library(dplyr)

# Count occurrences of Sleep Duration and arrange in descending order
sleep_duration_counts <- train_df %>%
  count(`Sleep Duration`) %>%
  arrange(desc(n))  # Sort by count in descending order

# Print the sorted counts
print(sleep_duration_counts)


library(dplyr)

# Define valid sleep durations
valid_sleep_durations <- c(
    'Less than 5 hours', '1-2 hours', '2-3 hours', '3-4 hours', '4-5 hours',
    '5-6 hours', '6-7 hours', '7-8 hours', '8-9 hours', '9-11 hours', '10-11 hours',
    'More than 8 hours'
)

# Define mapping to numerical values
sleep_duration_mapping <- c(
    'Less than 5 hours' = 4,
    '1-2 hours' = 1.5,
    '2-3 hours' = 2.5,
    '3-4 hours' = 3.5,
    '4-5 hours' = 4.5,
    '5-6 hours' = 5.5,
    '6-7 hours' = 6.5,
    '7-8 hours' = 7.5,
    '8-9 hours' = 8.5,
    '9-11 hours' = 10,
    '10-11 hours' = 10.5,
    'More than 8 hours' = 9
)

# Replace invalid entries with NA and map valid entries to numerical values
train_df$`Sleep Duration` <- ifelse(train_df$`Sleep Duration` %in% valid_sleep_durations,
                                     train_df$`Sleep Duration`,
                                     NA)

# Use the mapping to convert valid entries to numerical values
train_df$`Sleep Duration` <- as.numeric(sleep_duration_mapping[train_df$`Sleep Duration`])

# Print the counts of Sleep Duration after mapping
print(table(train_df$`Sleep Duration`))


library(dplyr)

# Assuming valid_sleep_durations and sleep_duration_mapping are already defined

# Replace invalid entries with NA in test_df
test_df$`Sleep Duration` <- ifelse(test_df$`Sleep Duration` %in% valid_sleep_durations,
                                    test_df$`Sleep Duration`,
                                    NA)

# Map valid entries to numerical values
test_df$`Sleep Duration` <- as.numeric(sleep_duration_mapping[test_df$`Sleep Duration`])

# Print the counts of Sleep Duration after mapping
print(table(test_df$`Sleep Duration`))


library(dplyr)

# Count occurrences of Degree and arrange in descending order
degree_counts <- train_df %>%
  count(Degree) %>%
  arrange(desc(n))  # Sort by count in descending order

# Print the sorted counts
print(degree_counts)

library(dplyr)

# Define valid degrees
valid_degrees <- c(
    "BHM", "LLB", "B.Pharm", "BBA", "MCA", "MD", "BSc", "ME", "B.Arch",
    "BCA", "BE", "MA", "B.Ed", "B.Com", "MBA", "M.Com", "MHM", "BA",
    "Class 12", "M.Tech", "PhD", "M.Ed", "MSc", "B.Tech", "LLM", "MBBS",
    "M.Pharm", "MPA", "BEd", "B.Sc", "M.Arch", "BArch", "Class 11"
)

# Define the mapping
degree_mapping <- c(
    "B.Sc" = "BSc", "B.Sc." = "BSc", "BEd" = "B.Ed", "M.Tech" = "M.Tech",
    "MSc" = "MSc", "PhD" = "PhD", "MEd" = "M.Ed", "B.Tech" = "B.Tech",
    "BE" = "B.E.", "B.Arch" = "B.Arch", "M.Com" = "M.Com", "B.Com" = "B.Com",
    "BHM" = "BHM", "LLB" = "LLB", "BA" = "BA", "MBA" = "MBA", "M.Arch" = "M.Arch"
)

# Replace values using the mapping
train_df$Degree <- recode(train_df$Degree, !!!degree_mapping)

# Replace invalid entries with NA
train_df$Degree <- ifelse(train_df$Degree %in% valid_degrees, train_df$Degree, NA)

# Print the counts of Degree after mapping
print(table(train_df$Degree))

library(dplyr)

# Assuming degree_mapping and valid_degrees are already defined

# Replace values using the mapping in test_df
test_df$Degree <- recode(test_df$Degree, !!!degree_mapping)

# Replace invalid entries with NA
test_df$Degree <- ifelse(test_df$Degree %in% valid_degrees, test_df$Degree, NA)

# Print the counts of Degree in test_df after mapping
print(table(test_df$Degree))


# ==============================================================================
# 1 Missing Data
# ==============================================================================

# Count the number of missing values in each column of train_df
missing_values <- colSums(is.na(train_df))

# Print the counts of missing values
print(missing_values)



# Load necessary library
library(dplyr)

# Select only numeric columns
numeric_cols_train <- train_df %>% select_if(is.numeric)

# Check the structure of the selected columns
print(numeric_cols_train)



# Define a threshold for NA removal (e.g., 50% NA values)
na_threshold <- 0.5 * nrow(train_df)

# Remove columns with more than the specified threshold of NAs
numeric_cols_train <- numeric_cols_train %>% select(where(~ sum(is.na(.)) < na_threshold))

# Check remaining numeric columns
print(numeric_cols_train)




# Check if there are any remaining numeric columns
if (ncol(numeric_cols_train) > 0) {
    # Calculate the correlation matrix
    correlation_matrix_train <- cor(numeric_cols_train, use = "complete.obs")
    
    # Print the correlation matrix
    print(correlation_matrix_train)
} else {
    print("No valid numeric columns available for correlation analysis.")
}



# Load libraries
library(reshape2)
library(ggplot2)

# If you want to use data.table's melt, you can do this:
# library(data.table)

# Reshape the correlation matrix for visualization
correlation_melted <- melt(correlation_matrix_train)

# Create the heatmap
ggplot(data = correlation_melted, aes(Var1, Var2, fill = value)) +
    geom_tile() +
    geom_text(aes(label = round(value, 2)), color = "white", size = 4) +
    scale_fill_gradient2(low = "blue", high = "yellow", mid = "green", 
                         midpoint = 0, limit = c(-1, 1), space = "Lab", 
                         name="Correlation") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Correlation Matrix of Numeric Features")
	
	
	
# ==============================================================================
# 2 p-value
# ==============================================================================	
	
	
# Load necessary library
library(dplyr)

# Updated list of categorical columns to check
categorical_cols <- c('Gender', 'City', 'Working Professional or Student', 
                       'Profession', 'Dietary Habits', 'Degree', 
                       'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness')

# Check for missing columns
missing_columns <- setdiff(categorical_cols, colnames(train_df))
if (length(missing_columns) > 0) {
    cat("The following columns are missing in train_df:\n")
    print(missing_columns)
}

# Loop through each categorical column to run chi-square test
for (col in categorical_cols) {
    if (col %in% colnames(train_df)) {
        # Filter out rows with NA values in the current column or the Depression column
        filtered_data <- train_df %>%
            filter(!is.na(get(col)), !is.na(Depression))
        
        # Create contingency table
        contingency_table <- table(filtered_data[[col]], filtered_data$Depression)
        
        # Perform chi-square test if contingency_table has more than one level
        if (nrow(contingency_table) > 1 && ncol(contingency_table) > 1) {
            chi_square_result <- chisq.test(contingency_table)
            
            # Print p-value
            cat(paste(col, ": p-value =", chi_square_result$p.value), "\n")
        } else {
            cat(paste(col, ": Not enough levels for Chi-square test"), "\n")
        }
    } else {
        cat(paste(col, ": Column not found in train_df"), "\n")
    }
}



# Count non-NA values for each column in train_df
non_na_counts <- colSums(!is.na(train_df))

# Display the counts
print(non_na_counts)


# ==============================================================================
# 3. Cleaning Data
# ==============================================================================

# Check the existing column names
existing_columns <- colnames(train_df)

# List of columns to drop (update based on your existing columns)
columns_to_drop <- c("Academic Pressure", "CGPA", "City", "Degree", "Profession", "Study Satisfaction")

# Keep only the columns that exist in the DataFrame
columns_to_drop <- columns_to_drop[columns_to_drop %in% existing_columns]

# Drop specified columns from train_df and test_df
train_df <- train_df %>%
    select(-all_of(columns_to_drop))

test_df <- test_df %>%
    select(-all_of(columns_to_drop))

# Fill missing values in numerical columns with the mean
numeric_cols <- c('Work Pressure', 'Job Satisfaction', 'Financial Stress', 'Sleep Duration')

fill_na_with_mean <- function(df, col) {
    df[[col]] <- ifelse(is.na(df[[col]]), mean(df[[col]], na.rm = TRUE), df[[col]])
    return(df)
}

for (col in numeric_cols) {
    train_df <- fill_na_with_mean(train_df, col)
    test_df <- fill_na_with_mean(test_df, col)
}

# Function to fill missing values with the mode
fill_na_with_mode <- function(df, col) {
    mode_value <- as.character(names(sort(table(df[[col]]), decreasing = TRUE)[1]))
    df[[col]][is.na(df[[col]])] <- mode_value
    return(df)
}

# Fill missing values in 'Dietary Habits' with the mode
train_df <- fill_na_with_mode(train_df, "Dietary Habits")
test_df <- fill_na_with_mode(test_df, "Dietary Habits")

# Check for remaining missing values
remaining_na_counts_train <- colSums(is.na(train_df))
remaining_na_counts_test <- colSums(is.na(test_df))

# Display remaining missing values
print(remaining_na_counts_train)
print(remaining_na_counts_test)

# ==============================================================================
# 3 Distribution of Age
# ==============================================================================

# Load necessary libraries
library(ggplot2)

# Create a histogram of the Age distribution
age_plot <- ggplot(train_df, aes(x = Age)) +
    geom_histogram(aes(y = ..count..), bins = 15, fill = "lightblue", color = "black", alpha = 0.7) +
    labs(title = "Distribution of Age", x = "Age", y = "Frequency") +
    theme_minimal()

# Add a blue line and points to the histogram
age_plot + 
    geom_line(stat = "bin", aes(y = ..count..), bins = 15, color = "blue", size = 1, group = 1) +  # Ensure line is drawn
    geom_point(stat = "bin", aes(y = ..count..), bins = 15, color = "blue", size = 3)  # Points along the line
	
	
	
# ==============================================================================
# 4 Distribution of Financial_Stress
# ==============================================================================	

# Assuming you have a categorical variable, e.g. "Gender"
facet_plot <- ggplot(train_df, aes(x = `Financial Stress`)) +
    geom_histogram(bins = 5, fill = "lightgreen", color = "black", alpha = 0.7) +
    labs(title = "Distribution of Financial Stress by Gender", x = "Financial Stress", y = "Frequency") +
    facet_wrap(~ Gender) + 
    theme_minimal()

# Display the facet plot
print(facet_plot)

# ==============================================================================
# 5 Distribution of Work_or_Study_Hours
# ==============================================================================

# Load necessary libraries
library(ggplot2)
library(dplyr)

# Create a summary data frame (if not already created)
frequency_counts_by_gender <- data.frame(
  Gender = c(rep("Female", 7), rep("Male", 7)),
  Category = c("0", "2", "4", "6", "8", "10", "More than 10",
               "0", "2", "4", "6", "8", "10", "More than 10"),
  n = c(5375, 9013, 8418, 8950, 8221, 11772, 10695,
        6558, 11147, 9912, 10570, 10336, 14811, 13223)
)

# Create the bar plot
work_study_hours_plot <- ggplot(frequency_counts_by_gender, aes(x = Category, y = n, fill = Gender)) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = "Distribution of Work or Study Hours by Gender", 
         x = "Work or Study Hours", 
         y = "Frequency") +
    theme_minimal() +
    geom_text(aes(label = n), position = position_dodge(width = 0.9), vjust = -0.5)  # Add counts above bars

# Display the plot
print(work_study_hours_plot)


# ==============================================================================
# 6 Count plot for Gender
# ==============================================================================

# Load necessary libraries
library(ggplot2)
library(dplyr)

# Create a count plot for Gender
gender_count_plot <- ggplot(train_df, aes(x = Gender, fill = Gender)) +
    geom_bar(color = "black", alpha = 0.7) +
    labs(title = "Count of Gender", 
         x = "Gender", 
         y = "Count") +
    scale_fill_manual(values = c("Female" = "lightpink", "Male" = "lightblue")) +  # Custom colors
    theme_minimal() +
    theme(text = element_text(size = 14),  # Increase text size for better readability
          plot.title = element_text(hjust = 0.5, size = 16)) +  # Center title and increase size
    geom_text(stat = 'count', aes(label = ..count..), 
              position = position_stack(vjust = 0.5),  # Position text above the bars
              size = 5, color = "black")

# Display the plot
print(gender_count_plot)


# ==============================================================================
# 7 Count plot for Have_you_ever_had_suicidal_thoughts 
# ==============================================================================

# Load necessary libraries
library(ggplot2)
library(dplyr)

# Create a count plot for "Have you ever had suicidal thoughts ?", faceted by Gender
suicidal_thoughts_plot <- ggplot(train_df, aes(x = `Have you ever had suicidal thoughts ?`, fill = Gender)) +
    geom_bar(color = "black", alpha = 0.7, position = "dodge") +  # Bars side by side
    labs(title = "Count of Suicidal Thoughts by Gender", 
         x = "Suicidal Thoughts", 
         y = "Count") +
    scale_fill_manual(values = c("Female" = "lightpink", "Male" = "lightblue")) +  # Custom colors
    theme_minimal() +
    theme(text = element_text(size = 14),  # Increase text size
          plot.title = element_text(hjust = 0.5, size = 16)) +  # Center title
    geom_text(stat = 'count', aes(label = ..count..), 
              position = position_dodge(width = 0.9),  # Position text above bars
              vjust = -0.5, size = 5, color = "black")

# Display the plot
print(suicidal_thoughts_plot)


# ==============================================================================
# 8-Distribution of Sleep_Duration
# ==============================================================================

# Load necessary libraries
library(ggplot2)
library(dplyr)

# Create a new categorical variable for Sleep Duration
train_df <- train_df %>%
    mutate(Sleep_Duration_Category = cut(`Sleep Duration`, 
                                          breaks = seq(0, 12, by = 2), 
                                          include.lowest = TRUE, 
                                          labels = c("0-2", "2-4", "4-6", "6-8", "8-10", "10-12")))

# Create a bar plot of Sleep Duration by Gender
sleep_duration_plot <- ggplot(train_df, aes(x = Sleep_Duration_Category, fill = Gender)) +
    geom_bar(position = "dodge", color = "black", alpha = 0.7) +
    labs(title = "Distribution of Sleep Duration by Gender", 
         x = "Sleep Duration (Hours)", 
         y = "Frequency") +
    scale_fill_manual(values = c("Female" = "lightpink", "Male" = "lightblue")) +  # Custom colors
    theme_minimal() +
    theme(text = element_text(size = 14),  # Increase text size
          plot.title = element_text(hjust = 0.5, size = 16)) +  # Center title
    geom_text(stat = "count", aes(label = ..count..), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, size = 4, color = "black")  # Counts above bars

# Display the plot
print(sleep_duration_plot)

# Remove the Sleep_Duration_Category column
train_df <- train_df %>%
    select(-Sleep_Duration_Category)

# ==============================================================================
# 9-Age vs Depression
# ==============================================================================

# Load necessary libraries
library(ggplot2)
library(dplyr)

# Create age categories for better visualization
train_df <- train_df %>%
    mutate(Age_Category = cut(Age, breaks = seq(0, 100, by = 10), 
                              right = FALSE, 
                              labels = paste(seq(0, 90, by = 10), seq(10, 100, by = 10), sep = "-")))

# Create a bar plot for Age Categories vs. Depression
age_depression_bar_plot <- ggplot(train_df, aes(x = Age_Category, fill = Depression)) +
    geom_bar(position = "dodge", color = "black", alpha = 0.7) +  # Set position to dodge for side-by-side bars
    labs(title = "Age Distribution by Depression Status", 
         x = "Age Categories", 
         y = "Count") +
    scale_fill_manual(values = c("0" = "#66c2a5", "1" = "#fc8d62"), 
                      name = "Depression Status", 
                      labels = c("0" = "No Depression", "1" = "Depressed")) +  # Custom colors and labels
    theme_minimal(base_size = 15) +  # Increase base font size
    theme(plot.title = element_text(hjust = 0.5, size = 18),  # Center title
          panel.grid.major = element_line(color = "gray80"),  # Customize grid
          panel.grid.minor = element_blank()) +  # Remove minor grid lines
    geom_text(stat = "count", aes(label = ..count..), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, size = 4, color = "black")  # Add count labels above bars

# Display the plot
print(age_depression_bar_plot)

# Remove the Age_Category column
train_df <- train_df %>%
    select(-Age_Category)


# ==============================================================================
# 11-Financial_Stress vs Depression
# ==============================================================================

# Load necessary libraries
library(ggplot2)
library(dplyr)

# Calculate the mean Financial Stress by Depression status
financial_stress_summary <- train_df %>%
    group_by(Depression) %>%
    summarise(mean_financial_stress = mean(`Financial Stress`, na.rm = TRUE))

# Create a colored bar plot with gradients
financial_stress_plot_stylish <- ggplot(financial_stress_summary, aes(x = as.factor(Depression), y = mean_financial_stress, fill = mean_financial_stress)) +
    geom_bar(stat = "identity", color = "black", alpha = 0.8) +  # Create the bar plot
    scale_fill_gradient(low = "#66c2a5", high = "#fc8d62") +  # Gradient color based on mean financial stress
    labs(title = "Financial Stress vs Depression", 
         x = "Depression Status", 
         y = "Mean Financial Stress") +
    theme_minimal(base_size = 15) +  # Use a minimal theme
    theme(plot.title = element_text(hjust = 0.5, size = 18),  # Center title
          text = element_text(size = 14)) +  # Increase text size
    geom_text(aes(label = round(mean_financial_stress, 1)), vjust = -0.5, size = 5)  # Add labels above the bars

# Display the plot
print(financial_stress_plot_stylish)



# ==============================================================================
# 12-Gender vs Depression
# ==============================================================================

# Load necessary libraries
library(ggplot2)
library(dplyr)

# Create a count summary for Gender and Depression
count_summary <- train_df %>%
    group_by(Gender, Depression) %>%
    summarise(count = n()) %>%
    ungroup()

# Create a count plot for Gender vs. Depression
gender_depression_plot <- ggplot(count_summary, aes(x = Gender, y = count, fill = as.factor(Depression))) +
    geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.7) +  # Create the count plot
    geom_text(aes(label = count), position = position_dodge(width = 0.9), vjust = -0.5, size = 5) +  # Add counts above bars
    labs(title = "Gender vs Depression", 
         x = "Gender", 
         y = "Count") +
    scale_fill_manual(values = c("0" = "#66c2a5", "1" = "#fc8d62"),  # Custom colors for each depression status
                      labels = c("0" = "No Depression", "1" = "Depressed")) +  # Custom labels
    theme_minimal(base_size = 15) +  # Use a minimal theme
    theme(plot.title = element_text(hjust = 0.5, size = 18),  # Center title
          text = element_text(size = 14))  # Increase text size

# Display the plot
print(gender_depression_plot)


# ==============================================================================
# 13-Work_or_Study_Hours vs Depression
# ==============================================================================

# Load necessary libraries
library(ggplot2)
library(dplyr)

# Create a count summary for Work or Study Hours and Depression
count_summary <- train_df %>%
    group_by(`Work/Study Hours`, Depression) %>%
    summarise(count = n()) %>%
    ungroup()

# Create a count plot for Work or Study Hours vs. Depression
work_study_hours_plot <- ggplot(count_summary, aes(x = as.factor(`Work/Study Hours`), y = count, fill = as.factor(Depression))) +
    geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.7) +  # Create the count plot
    labs(title = "Work or Study Hours vs Depression", 
         x = "Work or Study Hours", 
         y = "Count") +
    scale_fill_manual(values = c("0" = "#66c2a5", "1" = "#fc8d62"),  # Custom colors for each depression status
                      labels = c("0" = "No Depression", "1" = "Depressed")) +  # Custom labels
    theme_minimal(base_size = 15) +  # Use a minimal theme
    theme(plot.title = element_text(hjust = 0.5, size = 18),  # Center title
          text = element_text(size = 14))  # Increase text size

# Display the plot
print(work_study_hours_plot)


# ==============================================================================
# 13-Sleep_Duration' vs Depression`
# ==============================================================================


# Load necessary libraries
library(ggplot2)
library(dplyr)

# Create a count summary for Sleep Duration and Depression
count_summary <- train_df %>%
    group_by(`Sleep Duration`, Depression) %>%
    summarise(count = n()) %>%
    ungroup()

# Create a count plot for Sleep Duration vs. Depression
sleep_duration_plot <- ggplot(count_summary, aes(x = as.factor(`Sleep Duration`), y = count, fill = as.factor(Depression))) +
    geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.7) +  # Create the count plot
    geom_text(aes(label = count), position = position_dodge(width = 0.9), vjust = -0.5, size = 5) +  # Add counts above bars
    labs(title = "Sleep Duration vs Depression", 
         x = "Sleep Duration (Hours)", 
         y = "Count") +
    scale_fill_manual(values = c("0" = "#66c2a5", "1" = "#fc8d62"),  # Custom colors for each depression status
                      labels = c("0" = "No Depression", "1" = "Depressed")) +  # Custom labels
    theme_minimal(base_size = 15) +  # Use a minimal theme
    theme(plot.title = element_text(hjust = 0.5, size = 18),  # Center title
          text = element_text(size = 14))  # Increase text size

# Display the plot
print(sleep_duration_plot)



# ==============================================================================
# 14-Preprocessing data
# ==============================================================================


# Load necessary libraries
library(dplyr)
library(scales)

# Select columns to scale
columns_to_scale <- c('Age', 'Work Pressure', 'Job Satisfaction', 'Sleep Duration', 'Work/Study Hours', 'Financial Stress')

# Standard scaling for training data
df_scaled_train <- train_df %>%
    select(all_of(columns_to_scale)) %>%
    as.data.frame() %>%
    scale() %>%
    as.data.frame()

# Min-Max scaling for training data
min_max_scaler <- function(df) {
    df_scaled <- as.data.frame(lapply(df, function(x) rescale(x, to = c(0, 1))))
    return(df_scaled)
}

df_normalized_train <- min_max_scaler(df_scaled_train)

# Standard scaling for test data
df_scaled_test <- test_df %>%
    select(all_of(columns_to_scale)) %>%
    as.data.frame() %>%
    scale() %>%
    as.data.frame()

# Min-Max scaling for test data using the same scaler as training data
df_normalized_test <- min_max_scaler(df_scaled_test)

# Set column names
colnames(df_normalized_train) <- columns_to_scale
colnames(df_normalized_test) <- columns_to_scale

# Display the scaled and normalized data
print(head(df_normalized_train))



# ==============================================================================
# 15-Convert Categorical Data into Numerical
# ==============================================================================



# Load necessary libraries
library(dplyr)
library(caret)

# Define categorical columns
categorical_columns <- c('Gender', 'Working Professional or Student', 'Have you ever had suicidal thoughts ?', 'Dietary Habits', 'Family History of Mental Illness')

# Convert categorical columns to factors (Label Encoding for binary and preparation for One-hot Encoding)
train_df <- train_df %>%
    mutate(across(all_of(categorical_columns), as.factor))

test_df <- test_df %>%
    mutate(across(all_of(categorical_columns), as.factor))

# One-hot Encoding
dummies_train <- dummyVars(~ ., data = train_df %>% select(all_of(categorical_columns)))
train_df <- cbind(train_df %>% select(-all_of(categorical_columns)), 
                  as.data.frame(predict(dummies_train, newdata = train_df)))

dummies_test <- dummyVars(~ ., data = test_df %>% select(all_of(categorical_columns)))
test_df <- cbind(test_df %>% select(-all_of(categorical_columns)), 
                 as.data.frame(predict(dummies_test, newdata = test_df)))

# Drop the first level to avoid multicollinearity (equivalent to drop_first=True in Python)
train_df <- train_df %>% select(-matches("\\.1$"))
test_df <- test_df %>% select(-matches("\\.1$"))

# Display the head of the transformed dataframes
print(head(train_df))





# Load necessary libraries
library(dplyr)

# Function to convert TRUE/FALSE to 1/0
convert_bool_to_int <- function(df) {
  df %>%
    mutate(across(where(is.logical), ~ if_else(.x, 1L, 0L)))
}

# Apply the function to both train_df and test_df
train_df <- convert_bool_to_int(train_df)
test_df <- convert_bool_to_int(test_df)

# Display the head of the transformed dataframes
print(head(train_df))


# ==============================================================================
# 15-predect
# ==============================================================================

# Load necessary libraries
library(dplyr)
library(caret)

# Selecting features and target variable
X_train <- train_df %>%
    select(-id, -Name, -Depression)

y_train <- train_df$Depression

# Split the training data into training and validation sets (80-20 split)
set.seed(42)  # For reproducibility
train_index <- createDataPartition(y_train, p = 0.8, list = FALSE)

X_train_split <- X_train[train_index, ]
X_val_split <- X_train[-train_index, ]
y_train_split <- y_train[train_index]
y_val_split <- y_train[-train_index]

# Display the dimensions of the split datasets to verify
print(dim(X_train_split))
print(dim(X_val_split))
print(length(y_train_split))
print(length(y_val_split))


# ==============================================================================
# 15-Model Building
# Random Forest Classifier
# ==============================================================================

# Load necessary libraries
library(randomForest)
library(caret)
library(e1071)  # For confusionMatrix
library(pROC)   # For ROC-AUC calculation

# Build the Random Forest model
set.seed(42)  # For reproducibility
rf_model <- randomForest(x = X_train_split, y = as.factor(y_train_split), ntree = 100)

# Make predictions on the validation set
y_val_pred_rf <- predict(rf_model, X_val_split, type = "response")

# Print classification report
conf_matrix <- confusionMatrix(y_val_pred_rf, as.factor(y_val_split))
print(conf_matrix)

# Calculate ROC-AUC score
y_pred_prob <- predict(rf_model, X_val_split, type = "prob")[,2]
roc_auc_rf <- roc(as.numeric(y_val_split) - 1, y_pred_prob)$auc
print(paste("Random Forest - ROC-AUC Score:", roc_auc_rf))


# ==============================================================================
# 16-Plot ROC curve  Random Forest Classifier
# ==============================================================================

# Load necessary libraries
library(randomForest)
library(caret)
library(e1071)  # For confusionMatrix
library(pROC)   # For ROC-AUC calculation
library(ggplot2) # For plotting

# Build the Random Forest model
set.seed(42)  # For reproducibility
rf_model <- randomForest(x = X_train_split, y = as.factor(y_train_split), ntree = 100)

# Make predictions on the validation set
y_val_pred_rf <- predict(rf_model, X_val_split, type = "response")

# Print classification report
conf_matrix <- confusionMatrix(y_val_pred_rf, as.factor(y_val_split))
print(conf_matrix)

# Calculate ROC-AUC score
y_pred_prob <- predict(rf_model, X_val_split, type = "prob")[,2]
roc_auc_rf <- roc(as.numeric(y_val_split) - 1, y_pred_prob)$auc
print(paste("Random Forest - ROC-AUC Score:", roc_auc_rf))

# Get the fpr, tpr, and thresholds
roc_obj <- roc(as.numeric(y_val_split) - 1, y_pred_prob)
fpr <- 1 - roc_obj$specificities
tpr <- roc_obj$sensitivities
thresholds <- roc_obj$thresholds

# Plot ROC curve
roc_df <- data.frame(fpr = fpr, tpr = tpr, thresholds = thresholds)

ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = 'blue') +
  geom_abline(intercept = 0, slope = 1, color = 'gray', linetype = 'dashed') +
  ggtitle('ROC Curve for RFC Model') +
  xlab('False Positive Rate') +
  ylab('True Positive Rate') +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  annotate("text", x = 0.5, y = 0.1, label = paste("AUC =", round(roc_auc_rf, 4)), color = "blue")
  
  
  

# ==============================================================================
# 17- Logistic Regression
# ==============================================================================

# Load necessary libraries
library(caret)
library(e1071)  # For confusionMatrix
library(pROC)   # For ROC-AUC calculation

# Build the Logistic Regression model
lr_model <- glm(y_train_split ~ ., data = X_train_split, family = binomial)

# Make predictions on the validation set
y_val_pred_lr <- predict(lr_model, newdata = X_val_split, type = "response")

# Convert probabilities to class labels
y_val_pred_lr_class <- ifelse(y_val_pred_lr > 0.5, 1, 0)

# Print classification report
conf_matrix <- confusionMatrix(as.factor(y_val_pred_lr_class), as.factor(y_val_split))
print(conf_matrix)

# Calculate ROC-AUC score
roc_auc_lr <- roc(as.numeric(y_val_split) - 1, y_val_pred_lr)$auc
print(paste("Logistic Regression - ROC-AUC Score:", roc_auc_lr))



# ==============================================================================
# 17- Plot ROC curve Logistic Regression
# ==============================================================================

# Load necessary libraries
library(caret)
library(e1071)  # For confusionMatrix
library(pROC)   # For ROC-AUC calculation
library(ggplot2) # For plotting

# Build the Logistic Regression model
lr_model <- glm(y_train_split ~ ., data = X_train_split, family = binomial)

# Make predictions on the validation set
y_val_pred_lr <- predict(lr_model, newdata = X_val_split, type = "response")

# Convert probabilities to class labels
y_val_pred_lr_class <- ifelse(y_val_pred_lr > 0.5, 1, 0)

# Print classification report
conf_matrix <- confusionMatrix(as.factor(y_val_pred_lr_class), as.factor(y_val_split))
print(conf_matrix)

# Calculate ROC-AUC score
roc_obj <- roc(as.numeric(y_val_split) - 1, y_val_pred_lr)
roc_auc_lr <- roc_obj$auc
print(paste("Logistic Regression - ROC-AUC Score:", roc_auc_lr))

# Get the fpr, tpr, and thresholds
fpr2 <- 1 - roc_obj$specificities
tpr2 <- roc_obj$sensitivities
thresholds2 <- roc_obj$thresholds

# Plot ROC curve
roc_df <- data.frame(fpr = fpr2, tpr = tpr2, thresholds = thresholds2)

ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = 'blue') +
  geom_abline(intercept = 0, slope = 1, color = 'gray', linetype = 'dashed') +
  ggtitle('ROC Curve for LR Model') +
  xlab('False Positive Rate') +
  ylab('True Positive Rate') +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  annotate("text", x = 0.5, y = 0.1, label = paste("AUC =", round(roc_auc_lr, 4)), color = "blue")
  
  
  
  
# ==============================================================================
# 18- XGBoost
# ==============================================================================

# Load necessary libraries
library(xgboost)
library(caret)
library(e1071)  # For confusionMatrix
library(pROC)   # For ROC-AUC calculation

# Convert training and validation data to xgb.DMatrix
dtrain <- xgb.DMatrix(data = as.matrix(X_train_split), label = y_train_split)
dval <- xgb.DMatrix(data = as.matrix(X_val_split), label = y_val_split)

# Build the XGBoost model
set.seed(42)  # For reproducibility
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1
)

xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)

# Make predictions on the validation set
y_val_pred_xgb <- predict(xgb_model, dval)

# Convert probabilities to class labels
y_val_pred_xgb_class <- ifelse(y_val_pred_xgb > 0.5, 1, 0)

# Print classification report
conf_matrix <- confusionMatrix(as.factor(y_val_pred_xgb_class), as.factor(y_val_split))
print(conf_matrix)

# Calculate ROC-AUC score
roc_auc_xgb <- roc(as.numeric(y_val_split) - 1, y_val_pred_xgb)$auc
print(paste("xgboost - ROC-AUC Score:", roc_auc_xgb))


# ==============================================================================
# 18- Gradient Boosting Machine (GBM)
# ==============================================================================

install.packages("gbm")
# Load necessary libraries
library(gbm)
library(caret)
library(e1071)  # For confusionMatrix
library(pROC)   # For ROC-AUC calculation

# Build the GBM model
set.seed(42)  # For reproducibility
gbm_model <- gbm(
  formula = y_train_split ~ .,
  data = X_train_split,
  distribution = "bernoulli",
  n.trees = 100,
  interaction.depth = 3,
  shrinkage = 0.1,
  cv.folds = 5
)

# Make predictions on the validation set
y_val_pred_gbm <- predict(gbm_model, newdata = X_val_split, n.trees = gbm.perf(gbm_model, method = "cv"), type = "response")

# Convert probabilities to class labels
y_val_pred_gbm_class <- ifelse(y_val_pred_gbm > 0.5, 1, 0)

# Print classification report
conf_matrix <- confusionMatrix(as.factor(y_val_pred_gbm_class), as.factor(y_val_split))
print(conf_matrix)

# Calculate ROC-AUC score
roc_auc_gbm <- roc(as.numeric(y_val_split) - 1, y_val_pred_gbm)$auc
print(paste("GBM - ROC-AUC Score:", roc_auc_gbm))



