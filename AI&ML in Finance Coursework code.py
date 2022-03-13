#!/usr/bin/env python
# coding: utf-8

# In[137]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[138]:


stocks = pd.read_excel("C:/Users/bhava/OneDrive/Desktop/SEM 2/AI&ML in Finance/1st coursework/AI_COURSEWORK DATA.xlsx",index_col= 'DATE')


# In[139]:


stocks.head


# In[140]:


stocks.tail


# In[141]:


# Plot the adjusted close price
stocks.plot(figsize=(10, 7))
# Define the label for the title of the figure
plt.title("Adjusted Close Price of stocks" % stocks, fontsize=16)
# Define the labels for x-axis and y-axis
plt.ylabel('Adj Close Price', fontsize=14)
plt.xlabel('Year', fontsize=14)
# Plot the grid lines
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
# Show the plot
plt.show()


# In[142]:


# Log of percentage change of all stocks in the list
stocks1 = stocks.loc[:, stocks.columns != "Date"]
print(stocks1)


# In[178]:


stocks1.columns


# In[143]:


# Log of percentage change of all stocks in the list
Ret= stocks.pct_change().apply(lambda x: np.log(1+x)) # CONVERT TO RETURN
Ret.head()


# In[144]:


Ret.plot(title='Stocks weekly returns',figsize=(15,10))


# In[145]:


Annual_Return = stocks1.mean()*52
print (Annual_Return)


# In[146]:


Annual_Risk = Ret.std()*(52)**(0.5)
print(Annual_Risk)


# In[147]:


Return_Risk = (Annual_Return/Annual_Risk)
print(Return_Risk)


# In[181]:


Return_Risk.to_excel(r'C:/Users/bhava/OneDrive/Desktop/SEM 2/AI&ML in Finance/1st coursework/ReturnRisk.xlsx')


# In[148]:


stocks1.columns


# In[183]:


assets = pd.concat([Annual_Return,Annual_Risk,Return_Risk], axis=1)
assets.columns = ['Annual Returns', 'Annual Risk', 'Return Risk Ratio']
assets


# In[182]:


assets.to_excel(r'C:/Users/bhava/OneDrive/Desktop/SEM 2/AI&ML in Finance/1st coursework/Return-Risk Ratio.xlsx')


# In[150]:


x=Annual_Risk
y=Annual_Return
plt.scatter(x,y)
plt.show()


# In[151]:


Ret_Corr = Ret.corr()


# In[ ]:





# In[152]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(Ret_Corr)


# In[153]:


Ret_Covar = Ret.cov()
print(Ret_Covar)


# In[154]:


w = {'PFGDAAU LX Equity':0.043,'PFBKAAU LX Equity':0.043,'PFGOAAU LX Equity':0.043,'AIBALMA AB Equity':0.043,'CCASORI MK Equity':0.043,'BCUSDLF KK Equity':0.043,'ALPCRDA LX Equity':0.043,'PFGLAAU LX Equity':0.043, 'BBFTOP3 KY Equity':0.043, 'CCASLTF MK Equity':0.043,'LODFUIA LX Equity':0.043,'PFARAAU LX Equity':0.043,'GMFDEBI LE Equity':0.043,'UBSFLBI LX Equity':0.043, 'ALCONGP AB Equity':0.043, } 
port_var = Ret_Covar.mul(w, axis=0).mul(w, axis=1).sum().sum()
Annual_port_std = (port_var*52)**(0.5)
print (port_var)
print (port_var*52)


# In[155]:


Annual_port_std


# In[160]:


w = [0.043,0.043,0.043, 0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043,0.043]
port_er = (w*Annual_Return).sum()
port_er


# In[163]:


assets = pd.concat([Annual_Return,Annual_Risk,Return_Risk], axis=1)
assets.columns = ['Ann Returns', 'Ann Risk', 'Return Risk Ratio']
assets


# In[165]:


import matplotlib.pyplot as plt
x = Annual_Risk 
y = Annual_Return
plt.scatter(x, y)
plt.show()


# In[166]:


p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(stocks1.columns)
num_portfolios = 5000


# In[167]:


for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights,Annual_Return) 
    p_ret.append(returns)
    var = Ret_Covar.mul(weights, axis=0).mul(weights, axis=1).sum().sum() # Portfolio Variance
    sd = np.sqrt(var) #Daily standard deviation
    ann_sd = sd*np.sqrt(52) #Annual standard deviation = volatility
    p_vol.append(ann_sd)


# In[168]:


data = {'Returns':p_ret, 'Risk':p_vol}
for counter, symbol in enumerate(stocks1.columns.tolist()):
#print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]
portfolios = pd.DataFrame(data)
portfolios.head() # Dataframe of the 10000 portfolios created


# In[169]:


portfolios.plot.scatter(x='Risk', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[5,5])


# In[170]:


min_vol_port = portfolios.iloc[portfolios['Risk'].idxmin()]
# idxmin() gives us the minimum value in the column specified.
min_vol_port


# In[173]:


plt.subplots(figsize=[6,5])


# In[174]:


plt.scatter(portfolios['Risk'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=200)


# In[175]:


# Finding the optimal portfolio
rf = 0.01 # risk factor
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Risk']).idxmax()]
optimal_risky_port


# In[176]:


plt.subplots(figsize=(5, 6))
plt.scatter(portfolios['Risk'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)


# In[177]:


portfolios.plot.scatter(x='Risk', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[5,5])


# In[ ]:




