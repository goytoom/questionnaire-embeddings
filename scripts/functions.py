import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import pearsonr, mode 
from scipy import stats
from numpy import nanmean

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

#Models
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#random state
randState = 0

#functions to chose data set and embeddings type:
# Choose data set
def chooseData(data = "BIG5"): #choose big5 or ipip ### dataset paths   
  folder = data.upper() + "/"
  data = data.lower() + "_"
  return folder, data

#Choose embeddings type
def chooseEmb(emb = "USE"): #choose big5 or ipip ### dataset paths   
  embeddings = "questions_embeddings_" + emb.upper() +  ".csv"
  save = "_" + emb.upper() +  ".csv"
  return embeddings, save

#choose prediction model
def predModel(nr=1,par=1, randState=0):
  if nr == 1:
    model = RidgeClassifier(alpha=par, random_state=randState)
    modName = "RidgeClass"
  elif nr==2:  
    model = KNeighborsClassifier(n_neighbors=par)
    modName = "KNN"
  elif nr==3:
    model = SVC(C=par, random_state=randState)
    modName = "SVC"
  elif nr==4:  
    model = KNeighborsRegressor(n_neighbors=par)
    modName = "KnnReg"
  elif nr == 0:
    model = Ridge(alpha=par, random_state=randState)
    modName = "Ridge"
  # print("running {}: ".format(modName))
  return model, modName 

#choose reversed or non reversed data
def getResponses(folder, data, R=1):
  if R == 1:
    res = "responses.csv"
    responses = pd.read_csv("../embeddings/"+folder+data+res, index_col=0).T #reversed
    savePath = "../results/"+folder+"reversed/"
    items_ids = responses.columns
    items = responses.iloc[0,:].values
    if "item" in responses or "item" in responses.index:
      responses = responses.drop(["item"], axis=0)
  else:
    res = "responses_nonReversed.csv"
    responses = pd.read_csv("../embeddings/"+folder+data+res, index_col=0).T #reversed
    responses.columns.name = ""
    savePath = "../results/"+folder+"nonReversed/"
    items_ids = responses.columns
    items = responses.iloc[0,:].values
    if "item" in responses or "item" in responses.index:
      responses = responses.drop(["item"], axis=0)

  return responses.astype(float), savePath, items, items_ids 

##### Accuracy Functions

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    a = a[~np.isnan(a)] #remove nan
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def corrUserBased(x,y):
  #calculate correlations and p value for every person, then average:
  results = pd.DataFrame(np.full(shape=(x.shape[0],3),fill_value=0),index = x.index,columns=["Correlation","p-value","L1 Loss"])
  
  #new function, much faster, #NOT FOR IPIP, #adjust for p value of averages
  corrVals = x.corrwith(y, 1, method=lambda x, y: np.round(pearsonr(x, y), 6)).fillna(0)
  results.iloc[:,:2] = corrVals.values.tolist()
  results.iloc[:,2] = np.absolute(x - y).sum(1)/x.shape[1] 
  results = results.fillna(0)
  t, pval_total = stats.ttest_1samp(results.Correlation, popmean=0) #get p-value of average correlation tested against 0, #change to make n decimals!
  mean, ci_lower, ci_upper = mean_confidence_interval(results.Correlation)
  return results, [results.Correlation.mean(), pval_total, ci_lower, ci_upper, t, results.shape[0]-2]

# accuracy (percentage of correct classifications)
def accuracy_constr(x,y, constructs_list): #do we report the construct wise accuracy? we could otherwise use a heatmap!
  metric_constr = pd.DataFrame(np.zeros(shape=(1,len(constructs_list))))  #Dataframe for construct accuracy
  questions = x.index

  y_pred = x.values.flatten()
  y_test = y.loc[questions,:].values.flatten()
  y_test = y_test[~pd.isnull(y_pred)]
  y_pred = y_pred[~pd.isnull(y_pred)]
  y_pred = y_pred[~pd.isnull(y_test)]
  y_test = y_test[~pd.isnull(y_test)]
  metric_total = np.round(sum(y_pred==y_test)/len(y_test),3)
  return metric_total #list(metric_constr.mean()), metric_total

def accuracy_constr_comparison(x, y, constructs_total):
  constructs_total = constructs_total[x.index] #reorder according to x
  constructs_list = constructs_total.drop_duplicates().values.flatten()
  metric_constr = pd.DataFrame(np.zeros(shape=(1,len(constructs_list))), columns=constructs_list)  #Dataframe for construct accuracy
  y_test = y.loc[x.index,:].values.flatten() #re-order y according to x
  
  for constr in range(1, len(constructs_list)+1): # get performance by construct
    constr_idx = np.where(y_test == constr)[0]
    # get construct items/predictions
    y_pred_constr = x.values.flatten()[constr_idx]
    y_test_constr = y_test[constr_idx]
    # check for missing values
    y_test_constr = y_test_constr[~pd.isnull(y_pred_constr)]
    y_pred_constr = y_pred_constr[~pd.isnull(y_pred_constr)]
    y_pred_constr = y_pred_constr[~pd.isnull(y_test_constr)]
    y_test_constr = y_test_constr[~pd.isnull(y_test_constr)]

    acc_constr = np.round(sum(y_pred_constr==y_test_constr)/len(y_test_constr),3)
    metric_constr.iloc[0, constr-1] = acc_constr

  y_pred = x.values.flatten()
  y_test = y_test[~pd.isnull(y_pred)]
  y_pred = y_pred[~pd.isnull(y_pred)]
  y_pred = y_pred[~pd.isnull(y_test)]
  y_test = y_test[~pd.isnull(y_test)]
  metric_total = np.round(sum(y_pred==y_test)/len(y_test),3)
  return metric_total, metric_constr #list(metric_constr.mean()), metric_total

def accuracy_keys(x,y, constructs_list):
  metric_constr = pd.DataFrame(np.zeros(shape=(1,len(constructs_list))))  #Dataframe for construct correlation

  y_pred = x.values.flatten()
  y_test = y.loc[x.index].values.flatten()
  y_test = y_test[~pd.isnull(y_pred)]
  y_pred = y_pred[~pd.isnull(y_pred)]
  y_pred = y_pred[~pd.isnull(y_test)]
  y_test = y_test[~pd.isnull(y_test)]
  metric_total = np.round(sum(y_pred==y_test)/len(y_test),3)
  return metric_total

def accuracy_keys_comparison(x,y, constructs_total):
  constructs_total = constructs_total[x.index] #reorder according to x
  constructs_list = constructs_total.drop_duplicates().values.flatten()
  metric_constr = pd.DataFrame(np.zeros(shape=(1,len(constructs_list))), columns=constructs_list)  #Dataframe for construct accuracy
  y_test = y[x.index].values.flatten() #re-order y according to x
  
  for constr in constructs_list: # get performance by construct
    constr_idx = np.where(constructs_total == constr)[0]
    # get construct items/predictions
    y_pred_constr = x.values.flatten()[constr_idx]
    y_test_constr = y_test[constr_idx]
    # check for missing values
    y_test_constr = y_test_constr[~pd.isnull(y_pred_constr)]
    y_pred_constr = y_pred_constr[~pd.isnull(y_pred_constr)]
    y_pred_constr = y_pred_constr[~pd.isnull(y_test_constr)]
    y_test_constr = y_test_constr[~pd.isnull(y_test_constr)]

    acc_constr = np.round(sum(y_pred_constr==y_test_constr)/len(y_test_constr),3)
    metric_constr.loc[0, constr] = acc_constr

  y_pred = x.values.flatten()
  y_test = y_test[~pd.isnull(y_pred)]
  y_pred = y_pred[~pd.isnull(y_pred)]
  y_pred = y_pred[~pd.isnull(y_test)]
  y_test = y_test[~pd.isnull(y_test)]
  metric_total = np.round(sum(y_pred==y_test)/len(y_test),3)
  return metric_total, metric_constr

### Import functions: embeddings & response data
def getEmbeddings(folder, data, embeddings, responses):
  #read in embeddings, order them like the response data and save in vector
  embeddings_df = pd.read_csv("../embeddings/" + folder + data +  embeddings,index_col=0)
  question_ids = responses.columns  #get IDs of questions answered

  # only get embeddings of questions having answers in matrix
  embeddings_df = embeddings_df.loc[question_ids,:]
  # save to vector 
  X = embeddings_df.values

  ################################ Preprocess input data::
  #Standardize for regression (wihtout PCA)
  X_stand = pd.DataFrame(StandardScaler().fit_transform(X),index=question_ids) #standardize embeddings
  #With PCA, for regression, 90% variance explained
  X_pca = pd.DataFrame(PCA(0.9, random_state=0).fit_transform(X_stand),index=question_ids) 

  return embeddings_df, X_stand, X_pca

def getData(m, responses, X_pca_stand, folder, data):
  
  question_ids = responses.columns
  data_q = X_pca_stand
  
  #choose parameter range
  if m==2 or m==4:  #KNN and KNN Regression
    list_par = [1,5,10,15,50] #parameter search
  else:  #Ridge, SVC
    list_par = [10**x for x in range(5)] #parameter search

  constructs_list = pd.read_csv("../embeddings/" + folder + data + "questions_text.csv", encoding = 'utf-8',index_col=0)
  constrAssigned = constructs_list.loc[data_q.index,:] #only take constructs/encodings of items with embeddings
  constructs_list = constrAssigned.construct.drop_duplicates().values
  data_q = data_q.loc[constrAssigned.index,:] #only keep items that have an assigned construct

  return data_q, constructs_list, list_par, constrAssigned

def compareModels(verbose = 0):
    for l in ['sentencebert', "word2vec", "liwc"]: #go through the three embeddings
      for k in range(5):                       # calculate for each prediction model: Ridge, 
          #model:
          m = k            #0: Ridge, #1: RidgeClass, #2:KNN, #3: Kernel SVM (RBF), #4: KNN regression
          
          #data:
          R = 2            #1: reversed-coded, #2: nonReversed-coded
          d = "BIG5"       #data sets:   # BIG5, IPIP (all items), RIASEC, HSQ, 16PF
          e = l
          
          ############################ choose output here ##############################
          Path("../results/" + d.upper() + "/nonReversed/").mkdir(parents=True, exist_ok = True)
          Path("../results/" + d.upper() + "/reversed/").mkdir(parents=True, exist_ok = True)

          #####################################################################################

          #choose data set, embeddings, encoding
          folder, data = chooseData(d)        # BIG5, IPIP (all items), RIASEC, HSQ, 16PF
          embeddings, save = chooseEmb(e)     #USE, BERT, SENTENCEBERT
          responses, savePath, items, _ = getResponses(folder, data, R)
          X, X_stand, X_pca_stand = getEmbeddings(folder, data, embeddings, responses)

          #get embeddings name:
          embName = embeddings.split("_")[2].split(".")[0]

          # import required data and labels
          data_q, constructs_list, list_par, constrAssigned = getData(m, responses, X_pca_stand, folder, data)

          ############################## 10-Fold cross validation:
          #Split dataframe 10 fold
          kf = KFold(n_splits=10, random_state=randState, shuffle=True)
          questions = list(kf.split(data_q))

          #get ids
          question_ids = responses.columns  #get IDs of questions answered
          user_ids = responses.index        #get IDs of users
            
          ##################################### predictions #####################################

          for par in list_par:

            model, modelName = predModel(m,par, randState) 
            
            #Dataframes to store all predictions
            total_preds = pd.DataFrame(np.full(responses.shape, np.nan), columns=question_ids, index=user_ids)

            for q_fold, fold_nr in zip(questions,range(len(questions))): #go through the question folds
              #print("Fold {}:".format(str(fold_nr + 1))) #status/progress

              #train/test index and embeddings for current fold
              qid_train = q_fold[0]
              qid_test = q_fold[1]
              q_test = data_q.iloc[qid_test]

              # (train:) for each user go through all questions of the training fold
              for user,y in enumerate(responses.values):

                q_train = data_q.iloc[qid_train]
                #get responses for ith user on the training questions
                y_train = y[qid_train]
                q_train = q_train.loc[~pd.isnull(y_train),:]
                y_train = y_train[~pd.isnull(y_train)].astype('int')

                if (m==3) & (len(set(y_train)) == 1): #if only one type of response in training fold, use its value for prediction (because SVC does not work with only one class)
                  y_pred = np.repeat(y_train[0],q_test.shape[0])
                else: 
                  y_pred = np.round(model.fit(q_train,y_train).predict(q_test),0)

                #restrict to scale
                y_pred[y_pred < 1] = 1
                y_pred[y_pred > 5] = 5

                #save predictions in in dataframe
                total_preds.iloc[user, qid_test] = y_pred

            # calculate and print performance
            results, resultsMean = corrUserBased(total_preds, responses)
            corr, pval, ci_lower, ci_upper, t, dof = resultsMean
            
            if verbose==1:
                #Model metrics:
                print("MODEL: " + modelName + " (par = " + str(par) + ") " + embName + ":") #hand over string with specs...
                print("Correlation, lower-CI, upper-CI, p-value, t-statistic, DOF")
                print([round(corr,3), round(ci_lower,3), round(ci_upper,3), round(pval,3), round(t,3), round(dof,3)])
                print("\n")
            else:
                pass

            #save predictions
            total_preds.to_csv(savePath + modelName + "_" + str(par) + "_" + embName + "_responses.csv") #save predictions -> to calculate and predict performance for plots
    return 1

def modelPerformance(m=0, par=1, d="BIG5", e="sentencebert", verbose=0):
    
    # create directory to save results if not already
    Path("../results/" + d.upper() + "/nonReversed/").mkdir(parents=True, exist_ok = True)
    Path("../results/" + d.upper() + "/reversed/").mkdir(parents=True, exist_ok = True)
    
    # coding
    R = 2               #1: reversed, #2: nonReversed, RIASEC has only (j=2), IPIP with all items has only (j=2)

    #choose data set, embeddings, encoding
    folder, data = chooseData(d)                                      # BIG5, IPIP (all items), RIASEC, HSQ, 16PF
    embeddings, save = chooseEmb(e)                                   # USE, BERT, SENTENCEBERT
    responses, savePath, items, _ = getResponses(folder, data, R)
    X, X_stand, X_pca_stand = getEmbeddings(folder, data, embeddings, responses)

    #get embeddings name:
    embName = embeddings.split("_")[2].split(".")[0]

    # import required data and labels
    data_q, constructs_list, list_par, constrAssigned = getData(m, responses, X_pca_stand, folder, data)

    ############################## 10-Fold cross validation:
    #Split dataframe 10 fold
    kf = KFold(n_splits=10, random_state=randState, shuffle=True)
    questions = list(kf.split(data_q))

    #get ids
    question_ids = responses.columns  #get IDs of questions answered
    user_ids = responses.index        #get IDs of users

    #################################### predictions #####################################

    model, modelName = predModel(m,par) 
    #Dataframes to store all predictions
    total_preds = pd.DataFrame(np.full(responses.shape, np.nan), columns=question_ids, index=user_ids)
    total_dumb = pd.DataFrame(np.full(responses.shape, np.nan), columns=question_ids, index=user_ids)     #dumb predictor individual level

    for q_fold, fold_nr in zip(questions,range(len(questions))): #go through the question folds
      if verbose>1:
        print("Fold {}:".format(str(fold_nr + 1))) #status/progress
      else:
        pass
      #train/test index and embeddings for current fold
      qid_train = q_fold[0]
      qid_test = q_fold[1]
      q_test = data_q.iloc[qid_test]

      # (train:) for each userfold go through all questions of the question fold and concatenate question embedding to all users
      for user,y in enumerate(responses.values):

        q_train = data_q.iloc[qid_train]
        #get responses for ith user on the training questions
        y_train = y[qid_train]
        q_train = q_train.loc[~pd.isnull(y_train),:]
        y_train = y_train[~pd.isnull(y_train)].astype('int')

        if (m==3) & (len(set(y_train)) == 1): #if only one type of response in training fold, use value for prediction (SVC does not work with only one class)
          y_pred = np.repeat(y_train[0],q_test.shape[0])
        else: 
          y_pred = np.round(model.fit(q_train,y_train).predict(q_test),0)

        y_pred[y_pred < 1] = 1
        y_pred[y_pred > 5] = 5
        y_dumb = np.repeat(np.round(np.mean(y_train),0),q_test.shape[0]) 

        #save predictions in in dataframe
        total_preds.iloc[user, qid_test] = y_pred
        total_dumb.iloc[user, qid_test] = y_dumb

    results, resultsMean = corrUserBased(total_preds, responses)
    corr, pval, ci_lower, ci_upper, t, dof = resultsMean
    #naive baseline:
    results_dumb, resultsMean_dumb = corrUserBased(total_dumb, responses)
    corr_dumb, pval_dumb, ci_lower_dumb, ci_upper_dumb, t_dumb, dof_dumb = resultsMean_dumb
    
    if verbose>=1:
        print("MODEL: " + modelName + " (par = " + str(par) + ") " + embName + ":") #hand over string with specs...
        print("Correlation, lower-CI, upper-CI, p-value, t-statistic, DOF")
        print([round(corr,3), round(ci_lower,3), round(ci_upper,3), round(pval,3), round(t,3), round(dof,3)])
        print("\n")
    else:
        pass

    total_preds.to_csv(savePath + modelName + "_" + str(par) + "_" + embName + "_responses.csv")
    total_dumb.to_csv(savePath + 'baseline' + "_responses.csv")
    return 1

""" Human vs Model comparison (Analysis I) """
# calculate confidence interval around correlation
def pearsonr_ci(r, p, n, alpha=0.05):
    r_z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

# calculate confidence interval around mean
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    a = a[~np.isnan(a)] #remove nan
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# calculate correlations for model predictions and human raters
def regCorr(target_list, n_steps, model_data): #also calculate model data in the same way
    responses = pd.read_csv('../embeddings/' + "BIG5/" + "big5_responses_nonReversed.csv", index_col=0) #load actual participant responses

    corrs_list = []
    pvals_list = []
    comp_corrs_list = []
    comp_pvals_list = []
    for i, (df, df_name) in enumerate(target_list):
        y = responses.loc[:,df.columns[1]].values
        y = y[df.index]
        corrs = []
        pvals = []
        for col in df.iloc[:,2:]:
            full_col = df.loc[:,col]
            for nr_p in range(int(df.shape[0]/n_steps)):
                human_preds = full_col[nr_p*n_steps:(nr_p+1)*n_steps] #there are 10 predictions per person
                y_curr = y[nr_p*10:nr_p*10+10]
                y_curr = y_curr[~np.isnan(human_preds)] #remove nan
                x_curr = human_preds[~np.isnan(human_preds)]
                
                try: #for participants with not enough ratings!
                    cor, p = pearsonr(y_curr, x_curr)
                except:
                    cor = np.nan
                    p = np.nan
                corrs.append(cor)
                pvals.append(p)
              
        corrs_list.append(corrs)
        pvals_list.append(pvals)            
        
        #computational model's performance
        predictions = model_data.loc[model_data.index[i], model_data.columns[df.index]].values.reshape((10,10))
        actual_values = y.reshape((10,10)) #also check order -> import fold item order!
        comp_corrs = []
        comp_pvals = []
        for i in range(predictions.shape[0]):
            x_curr = predictions[i]
            y_curr = actual_values[i]
            
            try: #4 participants with not enough ratings!
                cor, p = pearsonr(y_curr, x_curr)
            except:
                cor = np.nan
                p = np.nan
            comp_corrs.append(cor)
            comp_pvals.append(p)
            
        comp_corrs_list.append(comp_corrs)
        comp_pvals_list.append(comp_pvals)

    return corrs_list, pvals_list, comp_corrs_list, comp_pvals_list

def welch_t_test(mu1, s1, N1, mu2, s2, N2):
  # Construct arrays to make calculations more succint.
  N_i = np.array([N1, N2])
  dof_i = N_i - 1
  v_i = np.array([s1, s2]) ** 2
  # Calculate t-stat, degrees of freedom, use scipy to find p-value.
  t = (mu1 - mu2) / np.sqrt(np.sum(v_i / N_i))
  dof = (np.sum(v_i / N_i) ** 2) / np.sum((v_i ** 2) / ((N_i ** 2) * dof_i))
  p = stats.distributions.t.sf(np.abs(t), dof) * 2
  return t, p, dof

""" Analysis II Human judges vs Model """
# function to compare target and fold wise, save correlations in array
def predictionPerformance(targets_data, human_data, model_data, df_folds):
    
    rows = []
    for target in targets_data.target_nr: #iterate over targets
        for fold in range(10): #iterate over folds
            test_items_idx = df_folds.loc[df_folds.fold_nr==fold+1, "test_items"].iloc[0]
            test_items_names = ["q" + str(x) for x in test_items_idx]
            true_x  = targets_data.loc[targets_data.target_nr == target, test_items_names].iloc[0]
            model_x = model_data.loc[model_data.target_nr == target, test_items_names].iloc[0]
            corr_model, p_model = pearsonr(true_x, model_x)
            rows.append([corr_model, p_model, target, fold+1, "Model"])
            if any((human_data.target == target) & (human_data.fold == fold+1)):
                human_x = human_data.loc[(human_data.target == target) & (human_data.fold == fold+1), test_items_names].iloc[0]
                nas = np.logical_or(np.isnan(true_x), np.isnan(human_x))   # in case nan are in vector
                corr_human, p_human = pearsonr(true_x[~nas], human_x[~nas])
                rows.append([corr_human, p_human, target, fold+1, "Human"])
            else:
                corr_human, p_human = [np.nan, np.nan]
                rows.append([corr_human, p_human, target, fold+1, "Human"])

    df_comparison = pd.DataFrame(rows, columns=["Correlation", "pvalue", "target", "fold", "Predictor"])
    return df_comparison

def targetComparison(targets_data, human_data, model_data, df_folds):
    targets_h = []
    targets_m = []
    for target in targets_data.target_nr: #iterate over targets
        innerlist_h = []
        innerlist_m = []
        for fold in range(10): #iterate over folds
            test_items_idx = df_folds.loc[df_folds.fold_nr==fold+1, "test_items"].iloc[0]
            test_items_names = ["q" + str(x) for x in test_items_idx]
            true_x  = targets_data.loc[targets_data.target_nr == target, test_items_names].iloc[0]
            model_x = model_data.loc[model_data.target_nr == target, test_items_names].iloc[0]
            corr_model, p_model = pearsonr(true_x, model_x)
            innerlist_m.append(corr_model)
            if any((human_data.target == target) & (human_data.fold == fold+1)):
                human_x = human_data.loc[(human_data.target == target) & (human_data.fold == fold+1), test_items_names].iloc[0]
                nas = np.logical_or(np.isnan(true_x), np.isnan(human_x))   # in case nan are in vector
                corr_human, p_human = pearsonr(true_x[~nas], human_x[~nas])
                innerlist_h.append(corr_human)
            else:
                corr_human, p_human = [np.nan, np.nan]
                innerlist_h.append(corr_human)
        targets_m.append(innerlist_m)
        targets_h.append(innerlist_h)

    return targets_h, targets_m

""" Study 4 (Construct and Key Prediction) """
#create functions:
#calculate accuracy for the construct or direction prediction tasks
def contextAccuracy(context="construct", i='Big5'):
  performance = pd.DataFrame(columns=["Dataset", "Target", "Model", "Accuracy"])
  #get the output data and path for respective context
  if context=="keys":
    m   = 1
    par = 10  
    model, modelName = predModel(m,par) 
    e = "sentencebert"
    ending = "_encodings_constrBased"
    #load path and necessary variables:
    folder, data = chooseData(i)        # BIG5, HSQ, 16PF
    embeddings, save = chooseEmb(e)     # LIWC, WORD2VEC, SENTENCEBERT
    savePath = "../results/" + folder + "nonReversed/"
    #get embeddings name:
    embName = embeddings.split("_")[2].split(".")[0]
    #import required data and labels:
    embeddings_df = pd.read_csv("../embeddings/" + folder + data + embeddings,index_col=0)
    constructs = pd.read_csv("../embeddings/" + folder + data + "questions_text.csv", encoding = 'utf-8',index_col=0)
    constrAssigned  = constructs.loc[embeddings_df.index,:] #only take constructs/encodings of items with embeddings
    constrAssigned  = constrAssigned[constrAssigned.construct != 'not assigned']
    constructs_list = constrAssigned.construct.drop_duplicates().values
    y = constrAssigned.encoding.replace([-1,1],[1,0])
    func = accuracy_keys
  elif context=="construct":
    ending = "_constructs"
    m   = 1
    par = 1000 
    model, modelName = predModel(m,par) 
    e = "sentencebert"
    #load path and necessary variables:
    folder, data     = chooseData(i)        # BIG5, IPIP2 (only assigned items), RIASEC, HSQ, 16PF
    embeddings, save = chooseEmb(e)     # LIWC, WORD2VEC, SENTENCEBERT
    savePath = "../results/" + folder + "nonReversed/"
    #get embeddings name:
    embName  = embeddings.split("_")[2].split(".")[0]
    #import required data and labels:
    embeddings_df   = pd.read_csv("../embeddings/" + folder + data +  embeddings,index_col=0)
    constructs      = pd.read_csv("../embeddings/" + folder + data + "questions_text.csv", encoding = 'utf-8',index_col=0)
    constrAssigned  = constructs.loc[embeddings_df.index,:] #only take constructs/encodings of items with embeddings
    constrAssigned  = constrAssigned[constrAssigned.construct != 'not assigned']
    constructs_list = constrAssigned.construct.drop_duplicates().values
    #dicts to convert number to constructs and vice versa
    y_constr = constrAssigned.construct
    constrNr = {f: n for n, f in enumerate(constructs_list, 1)} #change construct name to integer for the prediction models
    y = pd.DataFrame(np.asarray((list(map(constrNr.get, y_constr)))), index= y_constr.index, columns=["constructs"]) #list of labels converted to numbers (1 to n)    
    func = accuracy_constr
  
  # get predicted responses of chosen model:
  total_preds = pd.read_csv(savePath + modelName + "_" + str(par) + "_" + embName + ending + ".csv", index_col=0)
  total_preds.index = total_preds.index.map(str)
  total_preds = total_preds.astype(float)

  total_preds2 = pd.read_csv(savePath + modelName + "_" + str(par) + "_" + 'LIWC' + ending + ".csv", index_col=0)
  total_preds2.index = total_preds2.index.map(str)
  total_preds2 = total_preds2.astype(float)

  total_preds3 = pd.read_csv(savePath + modelName + "_" + str(par) + "_" + 'WORD2VEC' + ending + ".csv", index_col=0)
  total_preds3.index = total_preds3.index.map(str)
  total_preds3 = total_preds3.astype(float)

  total_dumb = pd.read_csv(savePath + modelName + "_" + str(par) + "_" + embName + ending + '_dumb' + ".csv", index_col=0)
  total_dumb.index = total_dumb.index.map(str)
  total_dumb = total_dumb.astype(float)

  #get performance metrict
  results  = func(total_preds, y, constructs_list)
  results2 = func(total_preds2, y, constructs_list)
  results3 = func(total_preds3, y, constructs_list)
  results_dumb = func(total_dumb, y, constructs_list)

  #combine in single frame
  performance.loc[len(performance)] = [i, context, "Best Model",  results]
  performance.loc[len(performance)] = [i, context, "LIWC",        results2]
  performance.loc[len(performance)] = [i, context, "WORD2VEC",    results3]
  performance.loc[len(performance)] = [i, context, "Baseline",    results_dumb]
  #combine correlations of all datasets in one frame
  return performance

#get performance for construct/direction prediction task by construct (prediction accuracy on all item belonging to that construct)
def contextAccuracy_constr(context="construct", i='Big5'):
  performance = pd.DataFrame(columns=["Dataset", "Target", "Model", "Construct", "Accuracy"])
  #get the output data and path for respective context
  if context=="keys":
    m = 1
    par = 10  
    model, modelName = predModel(m,par) 
    e = "sentencebert"
    ending = "_encodings_constrBased"
    #load path and necessary variables:
    folder, data = chooseData(i)        # BIG5, HSQ, 16PF
    embeddings, save = chooseEmb(e)     #USE, BERT, SENTENCEBERT
    savePath = "../results/" + folder + "nonReversed/"
    #get embeddings name:
    embName = embeddings.split("_")[2].split(".")[0]
    # import required data and labels:
    embeddings_df = pd.read_csv("../embeddings/" + folder + data +  embeddings,index_col=0)
    constructs = pd.read_csv("../embeddings/" + folder + data + "questions_text.csv", encoding = 'utf-8',index_col=0)
    constrAssigned = constructs.loc[embeddings_df.index,:] #only take constructs/encodings of items with embeddings
    constrAssigned = constrAssigned[constrAssigned.construct != 'not assigned']
    constructs_list = constrAssigned.construct.drop_duplicates().values
    y = constrAssigned.encoding.replace([-1,1],[1,0])
    func = accuracy_keys_comparison
  elif context=="construct":
    ending = "_constructs"
    m = 1
    par = 1000 
    model, modelName = predModel(m,par) 
    e = "sentencebert"
    #load path and necessary variables:
    folder, data = chooseData(i)        # BIG5, IPIP2 (only assigned items), RIASEC, HSQ, 16PF
    embeddings, save = chooseEmb(e)     #USE, BERT, SENTENCEBERT
    savePath = "../results/" + folder + "nonReversed/"
    #get embeddings name:
    embName = embeddings.split("_")[2].split(".")[0]
    # import required data and labels:
    embeddings_df = pd.read_csv("../embeddings/" + folder + data +  embeddings,index_col=0)
    constructs = pd.read_csv("../embeddings/" + folder + data + "questions_text.csv", encoding = 'utf-8',index_col=0)
    constrAssigned = constructs.loc[embeddings_df.index,:] #only take constructs/encodings of items with embeddings
    constrAssigned = constrAssigned[constrAssigned.construct != 'not assigned']
    constructs_list = constrAssigned.construct.drop_duplicates().values
    # print(constructs_list)
    #dicts to convert number to constructs and vice versa
    constrNr = {f: n for n, f in enumerate(constructs_list, 1)} #change construct name to integer for the prediction models
    y = pd.DataFrame(np.asarray((list(map(constrNr.get, constrAssigned.construct)))), index= constrAssigned.construct.index, columns=["constructs"]) #list of labels converted to numbers (1 to n)    
    func = accuracy_constr_comparison
  
  # get predicted responses of chosen model:
  total_preds = pd.read_csv(savePath + modelName + "_" + str(par) + "_" + embName + ending + ".csv", index_col=0)
  total_preds.index = total_preds.index.map(str)
  total_preds = total_preds.astype(float)

  total_preds2 = pd.read_csv(savePath + modelName + "_" + str(par) + "_" + 'LIWC' + ending + ".csv", index_col=0)
  total_preds2.index = total_preds2.index.map(str)
  total_preds2 = total_preds2.astype(float)

  total_preds3 = pd.read_csv(savePath + modelName + "_" + str(par) + "_" + 'WORD2VEC' + ending + ".csv", index_col=0)
  total_preds3.index = total_preds3.index.map(str)
  total_preds3 = total_preds3.astype(float)

  total_dumb = pd.read_csv(savePath + modelName + "_" + str(par) + "_" + embName + ending + '_dumb' + ".csv", index_col=0)
  total_dumb.index = total_dumb.index.map(str)
  total_dumb = total_dumb.astype(float)

  #get performance metric
  performance = pd.DataFrame(columns=["Dataset", "Target", "Model", "Construct", "Accuracy"])
  results,  results_constr  = func(total_preds, y, constrAssigned.construct)
  results_constr[["Dataset", "Target", "Model"]] = [i, context, "Best Model"]
  results2, results_constr2 = func(total_preds2, y, constrAssigned.construct)
  results_constr2[["Dataset", "Target", "Model"]] = [i, context, "LIWC"]
  results3, results_constr3 = func(total_preds3, y, constrAssigned.construct)
  results_constr3[["Dataset", "Target", "Model"]] = [i, context, "WORD2VEC"]
  results_dumb, results_constr_dumb = func(total_dumb, y, constrAssigned.construct)
  results_constr_dumb[["Dataset", "Target", "Model"]] = [i, context, "Baseline"]

  results_constr_df = pd.melt(results_constr, id_vars= ["Dataset", "Target", "Model"], var_name='Construct', value_name='Accuracy')
  results_constr_df2 = pd.melt(results_constr2, id_vars= ["Dataset", "Target", "Model"], var_name='Construct', value_name='Accuracy')
  results_constr_df3 = pd.melt(results_constr3, id_vars= ["Dataset", "Target", "Model"], var_name='Construct', value_name='Accuracy')
  results_constr_df_dumb = pd.melt(results_constr_dumb, id_vars= ["Dataset", "Target", "Model"], var_name='Construct', value_name='Accuracy')
  
  #combine in single frame
  performance = pd.concat([results_constr_df, results_constr_df2, results_constr_df3, results_constr_df_dumb])

  #combine correlations of all datasets in one frame
  return performance

def compareModelsConstructs(verbose = 0):
    for  l in ['sentencebert', "word2vec", "liwc"]: #go through the three embeddings
      for k in range(1,4):                          #calculate for each prediction model

        d = "BIG5"             # BIG5, IPIP2 (only assigned items), RIASEC, HSQ, 16PF
        ################################################ model:
        m = k                  #1: RidgeClass, #2:KNN, #3: Kernel SVM (RBF)
        ################################################ choose data:
        e =  l                 #embeddings type: USE, BERT, SENTENCEBERT

        #import data set, embeddings, encoding
        folder, data = chooseData(d)        
        embeddings, save = chooseEmb(e)
        responses, savePath, items, _ = getResponses(folder, data, 2)
        X, X_stand, X_pca_stand = getEmbeddings(folder, data, embeddings, responses)

        #get embeddings name:
        embName = embeddings.split("_")[2].split(".")[0]

        # import required data and labels
        data_q, constructs_list, list_par, constrAssigned = getData(m, responses, X_pca_stand, folder, data)
        y_constr = constrAssigned.construct.values #get construct values
        #dicts to convert number to constructs and vice versa
        constrNr = {f: n for n, f in enumerate(constructs_list, 1)} #change construct name to integer for the prediction models
        y = pd.DataFrame(np.asarray((list(map(constrNr.get, y_constr)))), index= constrAssigned.index, columns=["constructs"]) #list of labels converted to numbers (1 to n)

        ##################################### predictions #####################################
        question_ids = data_q.index  #get IDs of questions answered

        #initialise parameters:
        kf_constr = KFold(n_splits=10, random_state=randState, shuffle=True) #10-Fold cross validation
        questions_constr = list(kf_constr.split(data_q))

        for par in list_par:
          model, modelName = predModel(m,par)  #model = LogisticRegression(C = par), 
          #Dataframes to store all predictions
          total_preds = pd.DataFrame(np.full((data_q.shape[0],1), np.nan), columns=["constructs"], index=question_ids)

          for q_fold, fold_nr in zip(questions_constr,range(len(questions_constr))): #go through the training folds      
            #train/test index and embeddings for current fold
            qid_train = q_fold[0]
            qid_test = q_fold[1]
            q_train = data_q.iloc[qid_train]
            q_test = data_q.iloc[qid_test]

            y_train = y.loc[q_train.index].values.flatten()
            y_pred  = np.round(model.fit(q_train, y_train).predict(q_test),0)
            #naive predictor uses mode of training data to predict the fold
            y_dumb = np.repeat(mode(y)[0],q_test.shape[0])

            #save predictions in in dataframe
            total_preds.iloc[qid_test,0] = y_pred

          #calculate model metrics:
          acc_total = accuracy_constr(total_preds, y, constructs_list) #correct classifications percentage
          #DUMB metrics:

          if verbose==1:
            #Model metrics:
            print("MODEL: " + modelName + " (par = " + str(par) + ") " + embName + ":") #hand over string with specs...
            print("Correct classifications across all folds: ", acc_total)
            print("\n")
          else:
            pass

          #save results
          total_preds.to_csv(savePath + modelName + "_" + str(par) + "_" + embName + "_constructs.csv")
            
    return 1

def modelPerformanceConstruct(m=0, par=1, d="BIG5", e="sentencebert", verbose=0):

    #import data set, embeddings, encoding
    folder, data = chooseData(d)        # BIG5, IPIP2 (only assigned items), RIASEC, HSQ, 16PF
    embeddings, save = chooseEmb(e)     #USE, BERT, SENTENCEBERT
    responses, savePath, items, _ = getResponses(folder, data, 2)
    X, X_stand, X_pca_stand = getEmbeddings(folder, data, embeddings, responses)

    #get embeddings name:
    embName = embeddings.split("_")[2].split(".")[0]

    # import required data and labels
    data_q, constructs_list, list_par, constrAssigned = getData(m, responses, X_pca_stand, folder, data)

    y_constr = constrAssigned.construct.values #get construct values
    #dicts to convert number to constructs and vice versa
    constrNr = {f: n for n, f in enumerate(constructs_list, 1)} #change construct name to integer for the prediction models
    y = pd.DataFrame(np.asarray((list(map(constrNr.get, y_constr)))), index= constrAssigned.index, columns=["constructs"]) #list of labels converted to numbers (1 to n)

    ##################################### predictions #####################################

    question_ids = data_q.index  #get IDs of questions answered
    kf_constr = KFold(n_splits=10, random_state=randState, shuffle=True) #10 Fold cross validation
    questions_constr = list(kf_constr.split(data_q))

    model, modelName = predModel(m,par)  #model = LogisticRegression(C = par), 
    #Dataframes to store all predictions
    total_preds = pd.DataFrame(np.full((data_q.shape[0],1), np.nan), columns=["constructs"], index=question_ids)
    total_dumb = pd.DataFrame(np.full((data_q.shape[0],1), np.nan), columns=["constructs"], index=question_ids)     #naive predictor individual level

    for q_fold, fold_nr in zip(questions_constr,range(len(questions_constr))): #go through the question folds
      #train/test index and embeddings for current fold
      qid_train = q_fold[0]
      qid_test = q_fold[1]
      q_train = data_q.iloc[qid_train]
      q_test = data_q.iloc[qid_test]

      y_train = y.loc[q_train.index].values.flatten()
      y_pred  = np.round(model.fit(q_train, y_train).predict(q_test),0)
      #naive predictor uses mode of training data to predict the fold
      y_dumb = np.repeat(stats.mode(y.constructs)[0],q_test.shape[0])

      #save predictions in in dataframe
      total_preds.iloc[qid_test,0] = y_pred
      total_dumb.iloc[qid_test,0] = y_dumb

    #calculate model metrics:
    acc_total = accuracy_constr(total_preds, y, constructs_list) #correct classifications percentage
    #naive metrics:
    acc_dumb_total = accuracy_constr(total_dumb, y, constructs_list)  #correct naive classifications percentage

    if verbose==1:
        #Model metrics:
        print("MODEL: " + modelName + " (par = " + str(par) + ") " + embName + ":") #hand over string with specs...
        print("Correct classifications across all folds: ", acc_total)
        print("\n")
    else:
        pass

    total_preds.to_csv(savePath + modelName + "_" + str(par) + "_" + embName + "_constructs.csv")
    total_dumb.to_csv(savePath + modelName + "_" + str(par) + "_" + embName + "_constructs_dumb.csv")
    
    return 1

def compareModelsKey(verbose=0):
    for  l in ['sentencebert', "word2vec", "liwc"]:     #go through the three embeddings
      for k in range(1,4):                              #calculate for each prediction model
        m = k                                           #1: RidgeClass, #2: KNN, #3: Kernel SVM (RBF)

        #data:
        d = "BIG5"         # BIG5, 16PF, HSQ 
        e = l              # Embeddings: USE, BERT, SENTENCEBERT

        #import data set, embeddings, encoding
        folder, data = chooseData(d)        
        embeddings, save = chooseEmb(e) 
        responses, savePath, items, _ = getResponses(folder, data, 2)
        X, X_stand, X_pca_stand = getEmbeddings(folder, data, embeddings, responses)

        #get embeddings name:
        embName = embeddings.split("_")[2].split(".")[0]

        # import required data and labels
        data_q, constructs_list, list_par, constrAssigned = getData(m, responses, X_pca_stand, folder, data)
        keys = constrAssigned.encoding.replace([-1,1],[1,0])

        ############################## 10-Fold cross validation:
        #Split dataframe 10 fold
        kf_constr = KFold(n_splits=responses.shape[1], random_state=randState, shuffle=True)
        questions_constr = list(kf_constr.split(data_q))
        question_ids = data_q.index  #get IDs of questions answered

        ##################################### predictions #####################################

        for par in list_par:
          model, modelName = predModel(m,par)

          #Dataframes to store all predictions
          total_preds = pd.DataFrame(np.full((data_q.shape[0],1), np.nan), columns=["encoding"], index=question_ids)

          for q_fold, fold_nr in zip(questions_constr,range(len(questions_constr))): #go through the training folds
            #print("Fold {}:".format(str(fold_nr + 1))) #status/progress

            #train/test index and embeddings for current fold
            qid_train = q_fold[0]
            qid_test = q_fold[1]
            idx = np.where(constrAssigned.construct[qid_test].values == constrAssigned.construct[qid_train].values)[0]
            qid_train_constr = qid_train[idx]
            q_train = data_q.iloc[qid_train_constr]
            q_test = data_q.iloc[qid_test]
            y_train = keys.loc[q_train.index].values.flatten()
            try: 
                y_pred  = np.round(model.fit(q_train, y_train).predict(q_test),0)
            except:
                if fold_nr == 0:
                    print("ERROR with Model: " + modelName + ", par = " + str(par))
                else:
                    pass

            #save predictions in in dataframe
            total_preds.iloc[qid_test,0] = y_pred
            
          #calculate model metrics:
          acc_total = accuracy_keys(total_preds, keys, constructs_list) #correct classifications percentage

          if verbose==1:
            #Model metrics:
            print("MODEL: " + modelName + " (par = " + str(par) + ") " + embName + ":") #hand over string with specs...
            print("Correct classifications across all folds: ", acc_total)
            print("\n")
          else:
            pass

          # save results
          total_preds.to_csv(savePath + modelName + "_" + str(par) + "_" + embName + "_encodings_constrBased.csv")
    return 1

def modelPerformanceKey(m=1, par=10, d="BIG5", e="sentencebert", verbose=0):

    #import data set, embeddings, encoding
    folder, data = chooseData(d)        # BIG5, HSQ, 16PF
    embeddings, save = chooseEmb(e)     # USE, BERT, SENTENCEBERT
    responses, savePath, items, _ = getResponses(folder, data, 2)
    X, X_stand, X_pca_stand = getEmbeddings(folder, data, embeddings, responses)

    #get embeddings name:
    embName = embeddings.split("_")[2].split(".")[0]

    # import required data and labels
    data_q, constructs_list, list_par, constrAssigned = getData(m, responses, X_pca_stand, folder, data)
    keys = constrAssigned.encoding.replace([-1,1],[1,0])

    ############################## 10-Fold cross validation:
    #Split dataframe 10 fold
    kf_constr = KFold(n_splits=responses.shape[1], random_state=randState, shuffle=True)
    questions_constr = list(kf_constr.split(data_q))
    question_ids = data_q.index  #get IDs of questions answered

    ##################################### predictions #####################################

    model, modelName = predModel(m,par)

    #Dataframes to store all predictions
    total_preds = pd.DataFrame(np.full((data_q.shape[0],1), np.nan), columns=["encoding"], index=question_ids)
    total_dumb = pd.DataFrame(np.full((data_q.shape[0],1), np.nan), columns=["encoding"], index=question_ids)     #naive baseline

    for q_fold, fold_nr in zip(questions_constr,range(len(questions_constr))): #go through the training folds

      #train/test index and embeddings for current fold
      qid_train = q_fold[0]
      qid_test = q_fold[1]
      idx = np.where(constrAssigned.construct[qid_test].values == constrAssigned.construct[qid_train].values)[0]
      qid_train_constr = qid_train[idx]
      q_train = data_q.iloc[qid_train_constr]
      q_test = data_q.iloc[qid_test]

      y_train = keys.loc[q_train.index].values.flatten()
      
      try: 
          y_pred  = np.round(model.fit(q_train, y_train).predict(q_test),0)
          #naive predictor uses mode of training data to predict the fold
          y_dumb = np.repeat(mode(keys.values.flatten())[0],q_test.shape[0])
      except:
        print("ERROR with Model: " + model + ", par = " + par)

      #save predictions in in dataframe
      total_preds.iloc[qid_test,0] = y_pred
      total_dumb.iloc[qid_test,0] = y_dumb

    #calculate model metrics:
    acc_total = accuracy_keys(total_preds, keys, constructs_list) #correct classifications percentage
    #naive metrics:
    acc_dumb_total = accuracy_keys(total_dumb, keys, constructs_list)  #correct naive classifications percentage

    if verbose==1:
      #Model metrics:
      print("MODEL: " + modelName + " (par = " + str(par) + ") " + embName + ":") #hand over string with specs...
      print("Correct classifications across all folds: ", acc_total)
      print("\n")
    else:
      pass

    # save results
    total_preds.to_csv(savePath + modelName + "_" + str(par) + "_" + embName + "_encodings_constrBased.csv")
    total_dumb.to_csv(savePath + modelName + "_" + str(par) + "_" + embName + "_encodings_constrBased_dumb.csv")
    
    return 1

""" Auxiliary code for plotting """
def change_width(ax, new_value) :
  for patch in ax.patches :
      current_width = patch.get_width()
      diff = current_width - new_value

      # we change the bar width
      patch.set_width(new_value)

      # we recenter the bar
      patch.set_x(patch.get_x() + diff * .5)