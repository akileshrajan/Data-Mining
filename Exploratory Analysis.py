
# coding: utf-8

# ## Name: Akilesh R. Student ID.1001091662
# ## Name: Ashwin R. Student ID.1001098716 
# ## Name: Anirudh R. Student ID.1001051262
# ## Code Implementation based on Lecture from Saravanan Thirumuruganathan, University of Texas at Arlington
# 

# special IPython command to prepare the notebook for matplotlib
get_ipython().magic(u'matplotlib inline')

#Array processing
import numpy as np

#Data analysis, wrangling and common exploratory operations
import pandas as pd
from pandas import Series, DataFrame

#A sane way to get web data
import requests

#Packages for web scraping. No need to use both. Feel free to use one of them.
from pattern import web
from bs4 import BeautifulSoup

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns

#For some of the date operations
import datetime


# ### Part 2 and 3: Exploratory Analysis and Visualization
# ======================

fec_all = pd.read_csv('fec_2012_contribution_subset.csv', low_memory=False)
parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}
fec_all['party'] = fec_all.cand_nm.map(parties)

fec_all = fec_all[fec_all.contb_receipt_amt > 0]
fec = fec_all[fec_all.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]



# ### Task 1: Descriptive Statistics
# --------------------------------

# In[ ]:

#print the details of the data frame.
print "\nDetails of FEC data frame are: \n"
print fec.info(verbose=True, buf=None, max_cols=None) 

#finding the number of rows and columns in the data frame.
t1b_num_rows = len(fec)
t1b_num_cols = len(fec.columns)
print "\n\n\n#Rows=%s, #Columns=%s" % (t1b_num_rows, t1b_num_cols) 

# The only numeric data is 'contb_receipt_amt' which is the amount of contribution. 

print "\n\n\nDescriptive details of contb_receipt_amt is \n", fec['contb_receipt_amt'].describe()

# Let us now print the number of unique values for few columns

t1d_num_uniq_cand_id = fec.cand_id.nunique()

t1d_num_uniq_cand_nm = fec.cand_nm.nunique()
t1d_num_uniq_contbr_city = fec.contbr_city.nunique()
t1d_num_uniq_contbr_st = fec.contbr_st.nunique()

print "\n\n"
print "#Uniq cand_id = ", t1d_num_uniq_cand_id
print "#Uniq cand_num = ", t1d_num_uniq_cand_nm
print "#Uniq contbr_city = ", t1d_num_uniq_contbr_city
print "#Uniq contbr_st = ", t1d_num_uniq_contbr_st


# ###  Basic Filtering
# 

# In[ ]:

# How much contributions did Obama and Romney made.

t2a_tot_amt_obama = sum(fec['contb_receipt_amt'][fec.cand_nm=='Obama, Barack'])
t2a_tot_amt_romney = sum(fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'])
print "\nTotal Contribution for Obama is %s and for Romney is %s" % (t2a_tot_amt_obama, t2a_tot_amt_romney)

# How much contribution did folks from California, New York and Texas make totally (i.e. to both Obama and Romney).
t2b_tot_amt_CA = sum(fec['contb_receipt_amt'][fec.contbr_st == 'CA'])
t2b_tot_amt_NY = sum(fec['contb_receipt_amt'][fec.contbr_st == 'NY'])
t2b_tot_amt_TX = sum(fec['contb_receipt_amt'][fec.contbr_st == 'TX'])
print "\nTotal contributions from California is %s, New York is %s and Texas is %s" % (t2b_tot_amt_CA, t2b_tot_amt_NY, t2b_tot_amt_TX)


# How much money did folks from Texas made to BO and MR?
# How much money did folks from UT Arlington made to BO and MR?
t2c_tot_contr_tx_bo = sum(fec['contb_receipt_amt'][fec.contbr_st == 'TX'][fec.cand_nm == 'Obama, Barack'])
t2c_tot_contr_tx_mr = sum(fec['contb_receipt_amt'][fec.contbr_st == 'TX'][fec.cand_nm == 'Romney, Mitt'])
t2c_tot_contr_uta_bo = sum(fec['contb_receipt_amt'][fec.contbr_employer == 'UT ARLINGTON'][fec.cand_nm == 'Obama, Barack'])
t2c_tot_contr_uta_mr = sum(fec['contb_receipt_amt'][fec.contbr_employer == 'UT ARLINGTON'][fec.cand_nm == 'Romney, Mitt'])

print "\nFrom TX, BO got %s and MR got %s dollars" % (t2c_tot_contr_tx_bo, t2c_tot_contr_tx_mr)
print "From UTA, BO got %s and MR got %s dollars" % (t2c_tot_contr_uta_bo, t2c_tot_contr_uta_mr)

#Task 2d: How much did Engineers from Google gave to BO and MR.
# This task is a bit tricky as there are many variations: eg, SOFTWARE ENGINEER vs ENGINEER and GOOGLE INC. vs GOOGLE
t2d_tot_engr_goog_bo = sum(fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'][(fec.contbr_employer.str.contains("GOOGLE") == True) & (fec.contbr_occupation.str.contains("ENGINEER") == True)])
t2d_tot_engr_goog_mr = sum(fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'][(fec.contbr_employer.str.contains("GOOGLE") == True) & (fec.contbr_occupation.str.contains("ENGINEER") == True)])
print "\nFrom Google Engineers, BO got %s and MR got %s dollars" % (t2d_tot_engr_goog_bo, t2d_tot_engr_goog_mr)


# ### Basic Aggregation 
# --------------------------

# In[ ]:

# For each state, print the total contribution they made to both candidates. 
t3a_state_contr_both = fec['contb_receipt_amt'].groupby(fec.contbr_st).sum()
print "\n\nTotal contribution made to both candidates by each state are", t3a_state_contr_both

# Now let us limit ourselves to TX. For each city in TX, print the total contribution made to both candidates
t3b_tx_city_contr_both = fec['contb_receipt_amt'][fec.contbr_st == 'TX'].groupby(fec.contbr_city).sum()
print "\n\nTotal contribution made to both candidates by each city in TX are", t3b_tx_city_contr_both

#Now let us zoom into  Arlington, TX. For each zipcode in Arlington, print the total contribution made to both candidates
t3c_arlington_contr_both = fec['contb_receipt_amt'][fec.contbr_st == 'TX'][fec.contbr_city == 'ARLINGTON'].groupby(fec.contbr_zip).sum()
print "\n\nTotal contribution made to both candidates by each zipcode in Arlington are", t3c_arlington_contr_both


# ###  Aggregation+Filtering+Ranking
# -----------------------------------------



# Print the number of contributors to Obama in each state.
t4a_num_contr_obama_per_state = fec['contbr_nm'][fec.cand_nm == 'Obama, Barack'].groupby(fec.contbr_st).count()
print "\n\nNumber of contributions to Obama in each state is ", t4a_num_contr_obama_per_state

# Print the top-10 states (based on number of contributors) that contributed to Obama.
# print both state name and number of contributors
t4b_top10_obama_contr_states = fec['contbr_nm'][fec.cand_nm == 'Obama, Barack'].groupby(fec.contbr_st).count().order(ascending = False)[:10]
print "\n\nTop-10 states with most contributors to Obama are ", t4b_top10_obama_contr_states

# Print the top-20 occupations that contributed overall (to both BO and MR)
t4c_top20_contr_occupation = fec['contbr_nm'].groupby(fec.contbr_occupation).count().order(ascending = False)[:20]
print "\n\nTop-20 Occupations with most contributors are ", t4c_top20_contr_occupation

#Print the top-10 Employers that contributed overall (to both BO and MR)
t4d_top10_contr_employer_all = fec['contbr_nm'].groupby(fec.contbr_employer).count().order(ascending = False)[:10]
print "\n\nTop-10 Employers with most contributors are ", t4d_top10_contr_employer_all


# ### Basic Visualization
# -----------------------------



#Task 5a
task_5a = fec['contb_receipt_amt'].groupby(fec.cand_nm).sum().plot(kind='barh')
task_5a.set_xlabel('Total Amount Raised')
task_5a.set_ylabel('Candidate Names')
task_5a.set_title('5a')
plt.figure()

#Task 5b
task_5b = fec['contbr_nm'].groupby(fec.cand_nm).agg('count').plot(kind='barh')
task_5b.set_xlabel('Total Number of Contributors')
task_5b.set_ylabel('Candidate Names')
task_5b.set_title('5b')
plt.figure()

#Task 5c
task_5c = fec['contb_receipt_amt'].groupby(fec.cand_nm).mean().plot(kind='barh')
task_5c.set_xlabel('Average Contributions Made')
task_5c.set_ylabel('Candidate Names')
task_5c.set_title('5c')
plt.figure() 

#Task 5d
by_st = fec.pivot_table(values = 'contb_receipt_amt',index = ['contbr_st'],columns=['cand_nm'], aggfunc = np.sum)
by_st['Total'] = by_st.sum(axis=1)
by_st_sorted = by_st.sort(['Total'],ascending = False)[:10]
x = by_st_sorted.index
statelist=np.array(x)

obamalist = []
romneylist = []
for val in statelist:
    obama_contbr = (fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'][fec.contbr_st == val].sum()/fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'].sum()) *100
    romney_contbr = (fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'][fec.contbr_st == val].sum()/fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'].sum()) *100
    obamalist.append(float(obama_contbr))
    romneylist.append(float(romney_contbr))
o_arr = np.array(obamalist)
r_arr = np.array(romneylist)
width = 0.5
interval = np.arange(10)
graph,values = plt.subplots()
values.barh(interval, o_arr, width, color = 'blue')
values.barh(interval+0.5, r_arr, width, color = 'red')
values.set_title('5d')
values.set_xlabel('propotion of contribution the state made')
values.set_ylabel('Top 10 states')
values.legend(('Barack Obama','Mitt Romney'))
values.set_yticks(interval+0.5)
values.set_yticklabels(statelist)  

#Task 5e:
occ_list = []
by_occ = fec.pivot_table(values = 'contb_receipt_amt',index = ['contbr_occupation'],columns=['cand_nm'], aggfunc = np.sum)
by_occ['Total'] = by_occ.sum(axis=1)
by_occ_sorted = by_occ.sort(['Total'],ascending = False)[:10]
x = by_occ_sorted.index
occ_list=np.array(x)

obamalist = []
romneylist = []
for val in occ_list:
    obama_contbr = (fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'][fec.contbr_occupation == val].sum()/fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'].sum()) *100
    romney_contbr = (fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'][fec.contbr_occupation == val].sum()/fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'].sum()) *100
    obamalist.append(float(obama_contbr))
    romneylist.append(float(romney_contbr))
o_arr = np.array(obamalist)
r_arr = np.array(romneylist)
width = 0.5
interval = np.arange(10)
graph,values = plt.subplots()
values.barh(interval, o_arr,width, color = 'blue')
values.barh(interval+0.5, r_arr,width, color = 'red')
values.set_title('5e')
values.set_yticks(interval+width)
values.set_yticklabels(occ_list)
values.set_xlabel('propotion of contribution the occupation made')
values.set_ylabel('Top 10 Occupations')
values.legend(('Barack Obama','Mitt Romney'))

#Task 5f:
by_emp = fec.pivot_table(values = 'contb_receipt_amt',index = ['contbr_employer'],columns=['cand_nm'], aggfunc = np.sum)
by_emp['Total'] = by_emp.sum(axis=1)
by_emp_sorted = by_emp.sort(['Total'],ascending = False)[:10]
x = by_emp_sorted.index
employer_list=np.array(x)

obamalist = []
romneylist = []
for val in employer_list:
    obama_contbr = (fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'][fec.contbr_employer == val].sum()/fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'].sum()) *100
    romney_contbr = (fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'][fec.contbr_employer == val].sum()/fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'].sum()) *100
    obamalist.append(float(obama_contbr))
    romneylist.append(float(romney_contbr))
o_arr = np.array(obamalist)
r_arr = np.array(romneylist)
width = 0.5
interval = np.arange(10)
graph_emp,values_emp = plt.subplots()
values_emp.barh(interval, o_arr,width, color = 'blue')
values_emp.barh(interval+0.5, r_arr,width, color = 'red')
values_emp.set_title('5f')
values_emp.set_xlabel('propotion of contribution employers made')
values_emp.set_ylabel('Top 10 employers contributed')
values_emp.legend(('Barack Obama','Mitt Romney'))
values_emp.set_yticks(interval+width)
values_emp.set_yticklabels(employer_list)
plt.figure()

#Task 5g:
num_contbr = fec['contbr_nm'].groupby(fec.contbr_st).count()
task_g = num_contbr.plot(kind = 'bar')
task_g.set_xlabel('State Name')
task_g.set_ylabel('Number of Contributors')
task_g.set_title('5g')
plt.figure()

#Task 5h:
amt_bo = fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'].groupby(fec.contbr_st).count().hist(bins = 50)
amt_bo.set_xlabel('Contribution Bins')
amt_bo.set_ylabel('Frequency')
amt_bo.set_title('5h')
plt.figure()

#Task 5i:
amt_mr = fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'].groupby(fec.contbr_st).count().hist(bins = 50)
amt_mr.set_xlabel('Contribution Bins')
amt_mr.set_ylabel('Frequency')
amt_mr.set_title('5i')
plt.figure()

#Task 5j:
fec['contb_receipt_yr'] = pd.DatetimeIndex(fec['contb_receipt_dt']).year
fec['contb_receipt_quarter'] = pd.DatetimeIndex(fec['contb_receipt_dt']).quarter
contbr_quarter_bo = fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'].groupby([fec.contb_receipt_yr, fec.contb_receipt_quarter]).sum()
contbr_quarter_mr = fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'].groupby([fec.contb_receipt_yr, fec.contb_receipt_quarter]).sum()
task_j = contbr_quarter_bo.plot(kind = 'line')
task_j2 = contbr_quarter_mr.plot(kind = 'line', color = 'indianred')
task_j.set_xlabel('Yearly Quarter')
task_j.set_ylabel('Contribution amount')
task_j.set_title('5j')
task_j.legend(('Barack Obama','Mitt Romney'))
plt.figure()

#Task 5k:
contbr_quarter = fec.pivot_table(values = 'contb_receipt_amt',index = ['contb_receipt_yr','contb_receipt_quarter'],columns=['cand_nm'],fill_value = 0, aggfunc = np.sum)
contbr_quarter_cum = contbr_quarter.cumsum()
task_k = contbr_quarter_cum.plot(kind = 'line')
task_k.set_xlabel('Yearly Quarter')
task_k.set_ylabel('Contribution amount')
task_k.set_title('5k')
task_k.legend(('Barack Obama','Mitt Romney'))
plt.figure()

#Task 5l:
contbr_nm_quarter_bo = fec['contbr_nm'][fec.cand_nm == 'Obama, Barack'].groupby([fec.contb_receipt_yr, fec.contb_receipt_quarter]).count()
contbr_nm_quarter_mr = fec['contbr_nm'][fec.cand_nm == 'Romney, Mitt'].groupby([fec.contb_receipt_yr, fec.contb_receipt_quarter]).count()
task_l = contbr_nm_quarter_bo.plot(kind = 'line')
task_l2 = contbr_nm_quarter_mr.plot(kind = 'line', color = 'indianred')
task_l.set_xlabel('Yearly Quarter')
task_l.set_ylabel('Number of Contributors')
task_l.set_title('5l')
task_l.legend(('Barack Obama','Mitt Romney'))
plt.figure()


#Task 5m
contbr_nm_quarter = fec.pivot_table(values = 'contbr_nm',index = ['contb_receipt_yr','contb_receipt_quarter'],columns=['cand_nm'],fill_value = 0, aggfunc = np.size)
contbr_nm_quarter_cum = contbr_nm_quarter.cumsum()
task_m = contbr_nm_quarter_cum.plot(kind = 'line')
task_m.set_xlabel('Yearly Quarter')
task_m.set_ylabel('Contribution amount')
task_m.set_title('5m')
task_m.legend(('Barack Obama','Mitt Romney'))
plt.figure()


# ### Task 6: Discretization
# -----------------------

# In[ ]:

#Task 6a: Discretize the contributions of Obama and Romney based on the bins given below.

bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(fec['contb_receipt_amt'],bins) 


grouped = fec.groupby(['cand_nm',labels])
print "Task 6a:"
print grouped.size().unstack(0)

#Task 6b:
t6b_bucket_sums = fec['contb_receipt_amt'].groupby([fec.cand_nm,labels]).sum().unstack(0)
print "\nTask 6b:"
print t6b_bucket_sums


#Task 6c: 


t6c_normed_bucket_sums = t6b_bucket_sums.div(t6b_bucket_sums.sum(axis=1),axis=0)
print "\nTask 6c:"
print t6c_normed_bucket_sums

t6c = t6c_normed_bucket_sums[:-2].plot(kind='barh', stacked=True)
t6c.set_xlabel('normalization within bucket for both candidates')


#Task 6d: 
t6d_normed_bucket_sums = t6b_bucket_sums.div(t6b_bucket_sums.sum(axis=0),axis=1)
print "\nTask 6d:"
print t6d_normed_bucket_sums


t6d=t6d_normed_bucket_sums.plot(kind='barh', stacked=True)
t6d.set_xlabel('normalization within bucket per candidate')


# ### Task 7: Big Money in Politics
# ------------------------------

# In[ ]:

#Task 7a: 

def t7a_contributions_by_top_N_pct_obama(N):
    contb_obama = fec[fec.cand_nm == 'Obama, Barack']
    contb_table = pd.tools.pivot.pivot_table(contb_obama, values='contb_receipt_amt', index=['contbr_nm', 'contbr_occupation', 'contbr_employer', 'contbr_zip', 'contbr_city','contbr_st'], aggfunc=np.sum)
    sorted_table = contb_table.order(ascending=False)
    obama_tot_contb = sorted_table.sum()
    obamatbl_count = sorted_table.count()
    percent = int(N/100.0 *  obamatbl_count)
    topN_percent = sorted_table.iloc[:percent].sum()
    
    return topN_percent/obama_tot_contb
    
    
def t7a_contributions_by_top_N_pct_romney(N):
    contb_romney = fec[fec.cand_nm == 'Romney, Mitt']
    contb_table = pd.tools.pivot.pivot_table(contb_romney, values='contb_receipt_amt', index=['contbr_nm', 'contbr_occupation', 'contbr_employer', 'contbr_zip', 'contbr_city','contbr_st'], aggfunc=np.sum)
    sorted_table = contb_table.order(ascending=False)
    romney_tot_contb = sorted_table.sum()
    romneytbl_count = sorted_table.count()
    percent = int(N/100.0 *  romneytbl_count)
    topN_percent = sorted_table.iloc[:percent].sum()
    return topN_percent/romney_tot_contb

print "Task 7a:"
for N in [1, 2, 5, 10, 20]:
    print "N=%s, Obama proportion=%s and Romney proportion = %s" % (N, t7a_contributions_by_top_N_pct_obama(N), t7a_contributions_by_top_N_pct_romney(N))
    
    

#Task 7b: 

contb_table = pd.tools.pivot.pivot_table(fec, values='contb_receipt_amt', index=['contbr_nm', 'contbr_occupation', 'contbr_employer', 'contbr_zip', 'contbr_city','contbr_st'], aggfunc=np.sum)
tbl_count = contb_table.count()
percent = int(1/100.0 *  tbl_count)
#print percent
tbl_bycand = fec.pivot_table(values= 'contb_receipt_amt',index= ['contbr_nm', 'contbr_occupation', 'contbr_employer', 'contbr_city', 'contbr_st'],columns='cand_nm',fill_value=0,aggfunc='sum')
sorted_table = tbl_bycand.sort(['Obama, Barack','Romney, Mitt'],ascending = False)
t7b_1pcters = sorted_table.iloc[:percent]
print "\nTask 7b:"
print t7b_1pcters


# ### Task 8: Political Polarization in USA
# ------------------------------------------

# In[ ]:

#Task 8a: 
t8a_top_10_states_obama = fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'].groupby(fec.contbr_st).sum().order(ascending = False)[:10]
t8a_top_10_states_romney = fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'].groupby(fec.contbr_st).sum().order(ascending = False)[:10]
print "Task 8a\n"
print "Top 10 contributing states for Obama\n"
print t8a_top_10_states_obama
print "Top 10 contributing states for Romney\n"
print t8a_top_10_states_romney

#Task 8b:
t8b_top_10_occu_obama = fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'].groupby(fec.contbr_occupation).sum().order(ascending = False)[:10]
t8b_top_10_occu_romney = fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'].groupby(fec.contbr_occupation).sum().order(ascending = False)[:10]
print "\nTask 8b\n"
print "Top 10 contributing occupation for Obama\n"
print t8b_top_10_occu_obama
print "Top 10 contributing occupation for Romney\n"
print t8b_top_10_occu_romney



#Task 8c: 
t8c_top_10_emp_obama = fec['contb_receipt_amt'][fec.cand_nm == 'Obama, Barack'].groupby(fec.contbr_employer).sum().order(ascending = False)[:10]
t8c_top_10_emp_romney = fec['contb_receipt_amt'][fec.cand_nm == 'Romney, Mitt'].groupby(fec.contbr_employer).sum().order(ascending = False)[:10]
print "\nTask 8c\n"
print "Top 10 contributing employers for Obama\n"
print t8c_top_10_emp_obama
print "Top 10 contributing employers for Romney\n"
print t8c_top_10_emp_romney



#Task 8d: 
by_contribution = fec.pivot_table(values = 'contb_receipt_amt',index = ['contbr_nm','contbr_occupation','contbr_employer','contbr_city','contbr_st','contbr_zip'],columns=['cand_nm'], aggfunc = np.sum)
by_contribution['Total'] = by_contribution.sum(axis=1)
by_contbr_sorted = by_contribution.sort(['Total'],ascending = False)
by_contbr_flt = by_contbr_sorted[:1000]
by_contbr_flt.drop('Total',1,inplace = True)
romney_count = by_contbr_flt['Romney, Mitt'].count()
obama_count = by_contbr_flt['Obama, Barack'].count()
by_contbr_flt['check'] = by_contbr_flt.min(axis = 1)
by_contbr_flt.dropna(axis=0,how='any',inplace = True)
candidate_both = by_contbr_flt['check'].count()
t8d_top_1000_both = candidate_both
t8d_top_1000_BO_only = obama_count - candidate_both
t8d_top_1000_MR_only = romney_count - candidate_both
print "\nTask 8d\n"
print "Top 1000 Contributors to both:",t8d_top_1000_both
print "Top 1000 Contributors to Obama:", t8d_top_1000_BO_only
print "Top 1000 Contributors to Romney:", t8d_top_1000_MR_only


#Task 8e: 
by_state = fec.pivot_table(values= 'contb_receipt_amt',index= 'contbr_st',columns='cand_nm',fill_value = 0, aggfunc=np.sum)
by_state['Sorted_polarity'] = by_state.max(axis=1).div(by_state.min(axis=1)+1)
sorted_table = by_state.sort(['Sorted_polarity'],ascending = False) 
t8e_state_contr_ranked_by_polarity = sorted_table
print "\nTask 8e\n"
print "States ordered by decreasing polarity:"
print t8e_state_contr_ranked_by_polarity



#Task 8f: 
by_occupation = fec.pivot_table(values = 'contb_receipt_amt',index = 'contbr_occupation',columns='cand_nm',fill_value = 0, aggfunc = np.sum)
by_occupation_sorted = by_occupation.sort('Obama, Barack',ascending = False)[:50]
by_occupation_sorted['Sorted_polarity'] =  by_occupation_sorted.max(axis=1).div(by_occupation_sorted.min(axis=1)+1)
sorted_table_occupation = by_occupation_sorted.sort(['Sorted_polarity'], ascending = False)
t8f_occu_contr_ranked_by_polarity = sorted_table_occupation
print "\nTask 8f\n"
print "Top 50 occupation ordered by  polarity:"
print t8f_occu_contr_ranked_by_polarity



#Task 8g: 
by_city = fec.pivot_table(values = 'contb_receipt_amt',index = [fec.contbr_st == 'TX','contbr_city'],columns=['cand_nm'],fill_value = 0, aggfunc = np.sum)
by_city_tx = by_city.loc[True]
by_city_tx['Sorted_polarity'] =  by_city_tx.max(axis=1).div(by_city_tx.min(axis=1)+1)
sorted_table_city_tx = by_city_tx.sort(['Sorted_polarity'], ascending = False)
t8g_tx_city_contr_ranked_by_polarity = sorted_table_city_tx
print "\nTask 8g\n"
print "Cities in Texas ordered by decreasing polarity:"
print t8g_tx_city_contr_ranked_by_polarity

