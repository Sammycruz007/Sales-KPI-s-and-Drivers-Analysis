# Sales-KPI-s-and-Drivers-Analysis
# Transaction and Customer Data Analysis

## Project Overview

This project involves analyzing transaction and customer data to identify inconsistencies, clean the data, merge datasets, and assess trial vs. control store performance. The goal is to provide actionable recommendations for improving sales strategies.

## Motivation

The motivation behind this project is to apply the data science skills acquired during a course to solve a real-world data analytics problem. The dataset is sourced from a virtual internship platform and mimics an actual data analyst challenge from Quantium.

## Dataset Information

The dataset was obtained from a virtual internship platform and is designed to mimic a real-life data analyst problem from Quantium.

## Tools Used

- Python (scipy, statsmodels, scikit-learn, pandas, numpy, seaborn, matplotlib)
- Excel (for dashboard visualization)

## Project Tasks

1. **Transaction Data Examination**:

   - Identified inconsistencies, missing data, and outliers.
   - Ensured category items and numeric data were correctly classified.
   - Cleaned the dataset and saved the processed version.

2. **Customer Data Examination**:

   - Checked for missing values and inconsistencies.
   - Merged transaction and customer datasets.
   - Saved cleaned data for analysis.

3. **Control Store Selection**:

   - Defined metrics for control store selection.
   - Explored data and visualized key drivers.
   - Created a function to automate control store selection.

4. **Trial Store Performance Assessment**:

   - Analyzed each trial store’s performance against its control store.
   - Assessed the impact of interventions.
   - Determined the success of the trials.

5. **Findings & Recommendations**:

   - **Store 88**:
     - Investigate factors contributing to increased transactions per customer (e.g., promotions, product placement).
     - Apply successful strategies to other stores.
     - Monitor long-term customer retention.
   - **Stores 77 & 86**:
     - Reassess the trial design and intervention impact.
     - Consider extending the trial or implementing a different strategy (e.g., targeted marketing).
   - **Control Store Validation**:
     - Verify that control stores (233, 155, 237) matched trial stores in pre-trial performance.
     - Use correlation or distance metrics for validation.

## Bayesian Results

- Probability that Store 88's Total Sales are better: **100.00%**
- Probability that Store 88's Transactions per Customer are better: **54.48%**

## Insights from Dashboard (Excel Visualization)

- **Total Sales**: \$1.93M
- **Average Sales per Customer**: \$26.6
- **Total Customers**: 72.64K
- **Total Transactions**: 264.84K
- **Average Transactions per Customer**: 3.65
- **Top 3 Life-stage Sales Drivers**: Older Singles/Couples, Retirees, Older Families
- **Most Loyal Customer**: Customer ID **226000**, Total Purchase **1300**
- **Unique Number of Products Available**: 114
- **Total Products Sold**: 505,122
- **Top 5 Performing Stores**: 226, 88, 165, 40, 237
- **Customer Purchase Behavior by Sales**:
  - **Mainstream**: \$750.74K
  - **Budget**: \$676.21K
  - **Premium**: \$507.45K


## Business Questions Answered

Total Sales of $1.93M with an average of $26.6 per customer indicate a high-volume, low-ticket sales model.
The average of 3.65 transactions per customer suggests that most customers are making repeat purchases, which is a positive sign for customer engagement.

### Customer Engagement & Loyalty:

With 72.64K total customers driving 264.84K transactions, there's an opportunity to deepen relationships and increase the frequency or value of purchases.
The most loyal customer (ID 226000 with a total purchase of $1300) could provide insights into what drives high customer retention and spending, which can be leveraged to enhance loyalty programs.

### Product Dynamics:

Selling 505,122 units of products across 114 unique items indicates a strong product turnover. This suggests that maintaining an optimal inventory mix is crucial for meeting demand without overstocking.

### Store Performance:

The top 5 performing stores (226, 88, 165, 40, 237) serve as benchmarks. Analyzing their strategies could help replicate success in other locations.
Customer Purchase Behavior by Segment:

Mainstream: $750.74K in sales
Budget: $676.21K in sales
Premium: $507.45K in sales
This indicates that the mainstream and budget segments are the major revenue contributors, while the premium segment still represents a significant portion of sales.
Key Sales Drivers
Demographic Segments:

Top 3 Life-stage Sales Drivers: Older Singles/Couples, Retirees, and Older Families are the primary segments fueling sales. Tailoring marketing and product offerings to these groups can further drive growth.
Repeat Purchase Behavior:

The average transactions per customer (3.65) highlight that customer retention and repeat business are crucial. Programs to encourage loyalty and frequent shopping can have a significant impact.

### Store-Specific Performance:

High-performing stores (e.g., Store 226 and Store 88) likely have effective local strategies, customer engagement practices, or promotional tactics that can be modeled across other locations.

### Segment-Specific Revenue Generation:

The distribution of sales among mainstream, budget, and premium segments suggests a well-diversified customer base. This balance means that while the core revenue comes from mainstream and budget segments, there’s room to grow the premium segment through targeted offers and upselling strategies.The insights above help address key business questions on store performance, customer segmentation, and sales impact. 

## Conclusion

By applying data cleaning, statistical analysis, and visualization techniques, this project offers meaningful insights into trial store performance and customer behavior. The findings can guide strategic decision-making for optimizing sales and customer engagement.

