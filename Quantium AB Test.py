#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[31]:


# Load data (replace 'QVI_data.csv' with your file)
data = pd.read_csv('QVI_data.csv')
data.head(5)


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


data.columns.nunique()


# In[10]:


data.columns


# In[11]:


data.isnull().sum()


# In[34]:


# Convert DATE to datetime and extract year-month
data['DATE'] = pd.to_datetime(data['DATE'])
data['YEAR_MONTH'] = data['DATE'].dt.to_period('M')

# Aggregate monthly metrics for each store
monthly_data = data.groupby(['STORE_NBR', 'YEAR_MONTH']).agg(
    total_sales=('TOT_SALES', 'sum'),
    number_customers=('LYLTY_CARD_NBR', 'nunique'),
    total_transactions=('TXN_ID', 'count')
).reset_index()
monthly_data['avg_transactions_per_customer'] = monthly_data['total_transactions'] / monthly_data['number_customers']

monthly_data.head(5)


# #### Define Function to Find Control Stores

# In[49]:


# Using the Distance Method

def find_control_store1(trial_store, pre_trial_data, control_candidates):
    """
    Identify the best control store using Magnitude Distance.
    
    Parameters:
    - trial_store: The trial store number.
    - pre_trial_data: Pre-trial data containing metrics.
    - control_candidates: List of potential control stores.
    
    Returns:
    - The control store with the highest similarity score based on distance.
    """
    trial_metrics = pre_trial_data[pre_trial_data['STORE_NBR'] == trial_store]
    scores = []
    
    for control in control_candidates:
        control_metrics = pre_trial_data[pre_trial_data['STORE_NBR'] == control]
        merged = pd.merge(trial_metrics, control_metrics, on='YEAR_MONTH', suffixes=('_trial', '_control'))
        
        if merged.empty:
            continue
        
        # Calculate the magnitude distance for each metric
        distance = (
            (merged['total_sales_trial'] - merged['total_sales_control']).abs().sum() +
            (merged['number_customers_trial'] - merged['number_customers_control']).abs().sum() +
            (merged['avg_transactions_per_customer_trial'] - merged['avg_transactions_per_customer_control']).abs().sum()
        )
        
        # Normalize the distance and compute similarity score
        score = 1 - (distance - merged.shape[0]) / (merged.shape[0] * 3)  # Simplified normalization
        scores.append({'control_store': control, 'score': score})
    
    # Convert to DataFrame and return the best control store
    scores_df = pd.DataFrame(scores)
    return scores_df.loc[scores_df['score'].idxmax(), 'control_store']



# #### Identify control stores

# In[53]:


# Split into pre-trial and trial periods (adjust dates as needed)
pre_trial_data = monthly_data[monthly_data['YEAR_MONTH'] < pd.Period('2019-02', freq='M')]
trial_data = monthly_data[monthly_data['YEAR_MONTH'] >= pd.Period('2019-02', freq='M')]

trial_stores = [77, 86, 88]
control_candidates = [store for store in monthly_data['STORE_NBR'].unique() if store not in trial_stores]

# Find best control for each trial store
control_mapping1 = {}
for store in trial_stores:
    best_control1 = find_control_store1(store, pre_trial_data, control_candidates)
    control_mapping1[store] = best_control1
    print(f"Trial Store {store}: Best Control = {best_control1}")


# ####  Compare Performace

# In[55]:


# Comparing performance from the trial stores obtained by Correlation  Method

def compare_performance(trial_store, control_store, trial_data):
    trial_metrics = trial_data[trial_data['STORE_NBR'] == trial_store]
    control_metrics = trial_data[trial_data['STORE_NBR'] == control_store]
    
    # Perform t-tests
    sales_t, sales_p = ttest_ind(trial_metrics['total_sales'], control_metrics['total_sales'])
    customers_t, customers_p = ttest_ind(trial_metrics['number_customers'], control_metrics['number_customers'])
    trans_t, trans_p = ttest_ind(trial_metrics['avg_transactions_per_customer'], control_metrics['avg_transactions_per_customer'])
    
    print(f"\nTrial Store {trial_store} vs Control Store {control_store}:")
    print(f"Total Sales: p-value = {sales_p:.6f}")
    print(f"Customers: p-value = {customers_p:.6f}")
    print(f"Avg Transactions/Customer: p-value = {trans_p:.6f}")
    
    if sales_p < 0.05:
        print("\nSignificant difference in Total Sales!")
        
        if customers_p < 0.05:
            print("Driver: Change in Number of Customers")
          
        if trans_p < 0.05:
            print("Driver: Change in Transactions per Customer")
    print("."*50)

# Compare all pairs'
for trial_store, control_store in control_mapping1.items():
    compare_performance(trial_store, control_store, trial_data)


# Let’s analyze the results and determine the business implications:
# 
# ---
# 
# ### **1. Trial Store 77 vs Control Store 233**
# - **Total Sales:** `p = 0.3205` → **No significant difference.**
# - **Customers:** `p = 0.3431` → No significant change in customer count.
# - **Transactions/Customer:** `p = 0.4863` → No significant change in purchase frequency.
# 
# **Conclusion:**  
# The trial in Store 77 did not lead to a statistically significant improvement in sales, customer count, or purchase behavior compared to Control Store 233. The intervention likely had no measurable impact here.
# 
# ---
# 
# ### **2. Trial Store 86 vs Control Store 155**
# - **Total Sales:** `p = 0.3505` → **No significant difference.**
# - **Customers:** `p = 0.0533` → *Marginally significant* increase in customers (but still above 0.05).
# - **Transactions/Customer:** `p = 0.1161` → No significant change.
# 
# **Conclusion:**  
# Store 86’s trial showed a *slight hint* of attracting more customers, but the effect is not statistically conclusive. The intervention may need refinement or a longer trial period to validate this trend.
# 
# ---
# 
# ### **3. Trial Store 88 vs Control Store 237**
# - **Total Sales:** `p = 0.0191` → **Significant increase in sales!**  
# - **Customers:** `p = 0.1819` → No significant change in customer count.
# - **Transactions/Customer:** `p = 0.0103` → **Significant increase in purchase frequency!**
# 
# **Key Insight:**  
# The trial in Store 88 **succeeded** by driving existing customers to buy more frequently (e.g., upsells, cross-sells, or promotions encouraging repeat purchases). The increase in total sales was **not** due to attracting new customers but rather boosting engagement from existing ones.
# 
# ---
# 
# ### **Recommendations**
# 1. **For Store 88:**
#    - Investigate what drove the increase in transactions per customer (e.g., in-store promotions, product placement).
#    - Replicate successful strategies in other stores.
#    - Monitor long-term retention of these customers.
# 
# 2. **For Stores 77 & 86:**
#    - Re-evaluate the trial design (e.g., the intervention might not have been impactful enough).
#    - Consider extending the trial period or testing a different strategy (e.g., targeted marketing to attract new customers).
# 
# 3. **Control Store Validation:**
#    - Verify that the control stores (233, 155, 237) were well-matched to the trial stores in terms of pre-trial performance. Use your `find_control_store` function to check correlations or distances again.
# 
# 4. **Effect Size Analysis:**
#  
# 
# ---
# 
# ### **Why Did Store 88 Succeed?**
# Possible hypotheses to explore:
# - **In-Store Promotions:** Did Store 88 run a loyalty program or bundle deals?
# - **Staff Training:** Were employees trained to upsell/cross-sell?
# - **Product Availability:** Was there a popular product in Store 88 that was out of stock in the control store?
# 
# ---
# 
# ### **Next Steps**
# - **Root-Cause Analysis:** Interview Store 88’s manager to identify specific tactics used during the trial.
# - **Segmentation:** Use `LIFESTAGE` and `PREMIUM_CUSTOMER` columns to see if the increase was driven by a specific customer segment.
# - **Bayesian Analysis:** Confirm results using Bayesian methods to estimate the probability of success.
# 
# 

# #### Effect Size Analysis
# Calculate the magnitude of improvement for Store 88:

# In[56]:


# Example: Calculate % increase in total sales
trial_sales = trial_data[trial_data['STORE_NBR'] == 88]['total_sales'].mean()
control_sales = trial_data[trial_data['STORE_NBR'] == 237]['total_sales'].mean()
improvement = (trial_sales - control_sales) / control_sales * 100
print(f"Sales Improvement: {improvement:.2f}%")


# ###   Customer Segmentation Analysis
# We’ll analyze the LIFESTAGE and PREMIUM_CUSTOMER segments to identify high-performing groups.

# #### Filter Data for Trial Period

# In[57]:


# Define trial period (adjust dates if needed)
trial_start = '2019-02-01'
trial_end = '2019-04-30'

# Filter data for trial store 88 and control store 237 during the trial period
trial_data = data[
    (data['STORE_NBR'] == 88) & 
    (data['DATE'].between(trial_start, trial_end))
]

control_data = data[
    (data['STORE_NBR'] == 237) & 
    (data['DATE'].between(trial_start, trial_end))
]


# #### Segment by LIFESTAGE and PREMIUM_CUSTOMER

# In[59]:


def analyze_segments(df):
    # Group by segments and calculate key metrics
    return df.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).agg(
        total_sales=('TOT_SALES', 'sum'),
        num_customers=('LYLTY_CARD_NBR', 'nunique'),
        total_transactions=('TXN_ID', 'count')
    ).reset_index()

# Analyze segments for trial and control stores
trial_segments = analyze_segments(trial_data)
control_segments = analyze_segments(control_data)

# Merge to compare segments side-by-side
merged_segments = pd.merge(
    trial_segments, 
    control_segments, 
    on=['LIFESTAGE', 'PREMIUM_CUSTOMER'], 
    suffixes=('_trial', '_control'),
    how='outer'
).fillna(0)  # Replace missing segments with 0

# Calculate percentage differences
merged_segments['sales_lift_pct'] = (
    (merged_segments['total_sales_trial'] - merged_segments['total_sales_control']) / 
    merged_segments['total_sales_control'].replace(0, 1e-6) * 100  # Avoid division by zero
)

merged_segments['trans_per_customer_lift_pct'] = (
    (merged_segments['total_transactions_trial'] / merged_segments['num_customers_trial'] -
     merged_segments['total_transactions_control'] / merged_segments['num_customers_control']) /
    (merged_segments['total_transactions_control'] / merged_segments['num_customers_control'].replace(0, 1e-6)) * 100
)

# Show top segments by sales lift
top_segments = merged_segments.sort_values('sales_lift_pct', ascending=False).head(5)


# In[60]:


# Lets Visualise the result

plt.figure(figsize=(12, 6))
sns.barplot(
    x='sales_lift_pct', 
    y='LIFESTAGE', 
    hue='PREMIUM_CUSTOMER', 
    data=merged_segments.sort_values('sales_lift_pct', ascending=False).head(10)
)
plt.title("Top Segments by Sales Lift (%) in Store 88 vs. Control")
plt.xlabel("% Sales Lift")
plt.show()


# The above charts show that Mid_age singles/Couples, Retirees and older families are the key drivers of sales in Store 88, led by the premium and mainstream customers

# ## Bayesian Analysis

# We’ll use Bayesian methods to estimate the probability that Store 88’s performance is better than the control store for:
# 
# Total sales
# 
# Transactions per customer

# In[62]:


# Define trial and control period
trial_start = '2019-02-01'
trial_end = '2019-04-30'

# Filter data for trial and control stores during the trial period
trial_data = data[(data['STORE_NBR'] == 88) & (data['DATE'].between(trial_start, trial_end))]
control_data = data[(data['STORE_NBR'] == 237) & (data['DATE'].between(trial_start, trial_end))]

# Calculate metrics for Bayesian comparison
# Total Sales
sales_trial = trial_data['TOT_SALES'].sum()
sales_control = control_data['TOT_SALES'].sum()

# Transactions per Customer
txn_per_cust_trial = trial_data['TXN_ID'].nunique() / trial_data['LYLTY_CARD_NBR'].nunique()
txn_per_cust_control = control_data['TXN_ID'].nunique() / control_data['LYLTY_CARD_NBR'].nunique()

# Bayesian Beta-Binomial model for Total Sales
alpha_prior = 1
beta_prior = 1

# Total Sales (assumed successes/failures)
alpha_sales_trial = alpha_prior + sales_trial
beta_sales_trial = beta_prior + (100000 - sales_trial)  # Large cap to normalize

alpha_sales_control = alpha_prior + sales_control
beta_sales_control = beta_prior + (100000 - sales_control)  # Large cap to normalize

# Transactions per Customer
alpha_txn_trial = alpha_prior + txn_per_cust_trial
beta_txn_trial = beta_prior + (10 - txn_per_cust_trial)  # Normalize by scaling range

alpha_txn_control = alpha_prior + txn_per_cust_control
beta_txn_control = beta_prior + (10 - txn_per_cust_control)  # Normalize by scaling range

# Sample from posterior distributions
np.random.seed(42)
samples_sales_trial = np.random.beta(alpha_sales_trial, beta_sales_trial, 100000)
samples_sales_control = np.random.beta(alpha_sales_control, beta_sales_control, 100000)

samples_txn_trial = np.random.beta(alpha_txn_trial, beta_txn_trial, 100000)
samples_txn_control = np.random.beta(alpha_txn_control, beta_txn_control, 100000)

# Calculate probability of success
prob_sales_better = (samples_sales_trial > samples_sales_control).mean()
prob_txn_better = (samples_txn_trial > samples_txn_control).mean()

# Output results
print(f"Probability Store 88's Total Sales are better: {prob_sales_better:.2%}")
print(f"Probability Store 88's Transactions per Customer are better: {prob_txn_better:.2%}")


# 
# 
# 
# ### **Results Recap**
# 1. **Total Sales**:  
#    - **Probability: 100.00%**  
#    - This means Store 88 has a **very strong likelihood** of outperforming the control store in terms of **total sales** during the trial period.  
#    - The posterior distributions for **total sales** likely show little to no overlap between Store 88 and the control store, indicating a clear advantage.
# 
# 2. **Transactions per Customer**:  
#    - **Probability: 54.48%**  
#    - This suggests that Store 88 only has a **slightly better-than-random chance** of outperforming the control store in terms of **transactions per customer**.  
#    - The posterior distributions for **transactions per customer** likely have significant overlap, meaning there’s no clear evidence that Store 88 is consistently better in this metric.
# 
# ---
# 
# ### **Key Insights**
# 1. **Drivers of Higher Sales**:  
#    - Since **total sales** are significantly higher, the likely driver is either:
#      - **More customers making purchases**, or  
#      - **Higher transaction values per purchase**, rather than more frequent transactions per customer.  
#    - This aligns with the fact that **transactions per customer** shows only a modest difference.
# 
# 2. **Focus Areas**:  
#    - The results suggest that Store 88's success is **not strongly tied to customer purchasing frequency**.  
#    - Further analysis could focus on whether **higher spending per transaction** or **a larger customer base** is driving the increase in total sales.
# 
# ---
# 
# 

# In[ ]:




