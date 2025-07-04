# Airline Passenger Satisfaction Analysis using R

## 📊 Project Overview

This project analyzes airline passenger satisfaction data to identify key factors influencing passenger experience and develop predictive models for satisfaction levels. The analysis provides actionable insights for airlines to improve their services and enhance customer satisfaction.

## 🎯 Objectives

The primary goals of this analysis are to:

1. **Identify Key Drivers of Passenger Satisfaction**: Determine which factors (seat comfort, in-flight entertainment, customer service) most significantly influence passenger satisfaction ratings
2. **Analyze Flight-Related Impact**: Examine how flight duration, delays, and travel class affect passenger satisfaction
3. **Evaluate In-Flight Service Impact**: Assess the effects of Wi-Fi service, food & drink, seat comfort, entertainment, legroom, and cleanliness on satisfaction levels

## 📋 Dataset Description

The dataset contains comprehensive information about airline passenger satisfaction from various airlines, obtained from [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data).

### Key Features:
- **Demographics**: Age, Gender, Customer Type (Loyal/Disloyal)
- **Flight Details**: Flight Distance, Travel Class, Type of Travel
- **Service Ratings**: Wi-Fi, Food & Drink, Seat Comfort, Entertainment, Legroom, Cleanliness (Scale 1-5)
- **Operational Metrics**: Departure/Arrival Delays, Gate Location, Baggage Handling
- **Target Variable**: Satisfaction Level (Satisfied, Neutral, Dissatisfied)

## 🔍 Key Findings

### Satisfaction Drivers
- **Online Boarding** emerged as the most critical factor (root node in decision tree)
- **In-flight Wi-Fi Service** and **Type of Travel** are secondary drivers
- **Class of Travel** significantly impacts satisfaction levels

### Demographic Insights
- Age groups 23-27 and 42-47 are frequent flyers
- Male passengers show slightly higher satisfaction than female passengers
- Business travelers have significantly higher satisfaction (58.4%) compared to personal travelers (10.1%)

### Flight-Related Factors
- Economy and Economy Plus classes show much lower satisfaction compared to Business class
- Short-distance flights experience more departure and arrival delays
- Most passenger trips are domestic (< 1500 miles)

## 🛠️ Methodology

### Machine Learning Models Used:
1. **Decision Tree**: For identifying key satisfaction drivers (88.46% accuracy)
2. **Logistic Regression**: For flight-related factor analysis (79.9% accuracy)
3. **Naive Bayes Classifier**: For comparison with logistic regression (78.23% accuracy)

### Analysis Techniques:
- Exploratory Data Analysis (EDA) with various visualizations
- Feature importance analysis
- Correlation analysis
- Predictive modeling
- Model performance evaluation using confusion matrices and ROC curves

## 📈 Visualizations

The project includes comprehensive visualizations:
- **Histograms**: Age distribution, flight distance patterns
- **Bar Charts**: Gender, travel type, and class satisfaction percentages
- **Scatter Plots**: Flight distance vs. delays analysis
- **Box Plots**: Service ratings across different travel classes
- **Decision Trees**: Feature importance visualization
- **ROC Curves**: Model performance evaluation

## 💡 Managerial Insights & Recommendations

### Immediate Actions:
1. **Optimize Online Boarding**: Streamline check-in processes and provide clear instructions
2. **Enhance In-Flight Wi-Fi**: Invest in reliable, high-speed connectivity
3. **Improve Economy Class Services**: Focus on cleanliness, comfort, and basic amenities
4. **Address Short-Flight Delays**: Implement measures to reduce delays on domestic routes

### Strategic Initiatives:
1. **Class-Based Service Differentiation**: Tailor services based on travel class expectations
2. **Travel Type Customization**: Develop specific service packages for business vs. personal travelers
3. **Predictive Satisfaction Management**: Use models to proactively identify and address potential dissatisfaction

## 🚀 Getting Started

### Prerequisites
```r
# Required R packages
install.packages(c("ggplot2", "dplyr", "caret", "rpart", "randomForest", 
                   "ROCR", "e1071", "corrplot", "gridExtra"))
```

### Running the Analysis
1. Clone this repository
2. Ensure you have R and RStudio installed
3. Install required packages
4. Load the dataset from the provided Kaggle link
5. Run the R script to reproduce the analysis

## 📊 Model Performance

| Model | Accuracy | Use Case |
|-------|----------|----------|
| Decision Tree | 88.46% | Overall satisfaction prediction |
| Logistic Regression | 79.9% | Flight-related factor analysis |
| Naive Bayes | 78.23% | Baseline comparison |

## 🔮 Future Enhancements

1. **Real-time Prediction Dashboard**: Implement a web dashboard for real-time satisfaction prediction
2. **Advanced Feature Engineering**: Create new features from existing data
3. **Ensemble Methods**: Combine multiple models for improved accuracy
4. **Sentiment Analysis**: Incorporate text feedback analysis
5. **Time Series Analysis**: Study satisfaction trends over time

## 📝 License

This project is created for academic purposes. Dataset credit goes to the original Kaggle contributor.

## 🤝 Contributing

This is an academic project, but suggestions and feedback are welcome! Please feel free to:
- Open issues for bugs or suggestions
- Submit pull requests for improvements
- Share your insights and findings

## 📧 Contact

For questions or collaborations, please reach out to me via linkedin.

---

*This analysis demonstrates the power of data-driven decision making in the airline industry and provides actionable insights for improving passenger satisfaction.*
