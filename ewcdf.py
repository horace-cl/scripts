#function to create a sampler
def distribution_sampler(model,data):    
    angular_data=data.cosThetaKMu.to_numpy()
    sampler=model.create_sampler(len(angular_data))
    sampler_hist=[]
    for i in range (0,10000):
        sampler.resample()
        sampler_hist.append(sampler.to_pandas())
    statistic_sampler=[]
    for i in range (0,10000):
        statistic_sampler.append(ks_test(sampler_hist[i], model).statistic)
    return statistic_sampler
#function to plot the sampler and statistic line 
def histogram_distribution(distribution,statistic):
    histogram=(plt.hist(distribution,bins=50,color='teal'), 
               plt.title('statistic values from sampler'),
               plt.axvline(statistic,color='black'))
    return histogram    

  #function to find statistic with SNS 
  def ks_statistic_seaborn(data,model):
    #finding cdf 
    def cdf(ñ):
        return model.cdf(ñ) #model must specify degree
    #finding empirical cdf 
    plot_ewcdf_sns=sns.ecdfplot(data=data, x='cosThetaKMu', weights='totalW',color='darkblue')
    x_y_arrays=plot_ewcdf_sns.get_lines()[0].get_data()
    #finding heights
    heights_bins=[]
    for i in range (0,len(x_y_arrays[1])):
        heights_bins.append(x_y_arrays[1][i])

    heights_cdf=[]
    for i in range (0,len(x_y_arrays[0])):
        heights_cdf.append(cdf(np.array([-1,x_y_arrays[0][i]]))[1])

    #finding statistic 
    lenghts=[]
    for i in range (0,len(heights_bins)): #heights_cdf could also be used, both are supposed to have the same lenght
        lenghts.append(abs(heights_bins[i]-heights_cdf[i])) #assumig the values are ordered inside the list  
    
    statistic=max(lenghts) #Find Sup of the absolute values
    return statistic
  
  #primitive function to find satistic 
  def ks_statistic_emilia(data,model):
    #finding cdf
    def cdf(ñ):
        return model.cdf(ñ)
    #finding empirical cdf 
    weights=data.totalW.to_numpy()
    angular_data=data.cosThetaKMu.to_numpy()
    angular_data_ordered=np.sort(angular_data)

    def wecdf(y):
        n=angular_data.size
        total=0
        for i in range (0,n):
            if angular_data[i]<=y:
                total=total+weights[i]
        return total
    def wecdf_norm(y):
        return wecdf(y)/wecdf(1)
    heights_steps=[]
    for i in range (0,len(angular_data_ordered)):
        heights_steps.append(wecdf_norm(angular_data_ordered[i]))
    
    
    heights_cdf=[]
    for i in range (0,len(angular_data_ordered)):
        heights_cdf.append(cdf(np.array([-1,angular_data_ordered[i]]))[1])
 
    #finding statistic 
    distance=[]
    for i in range (0,len(heights_steps)): #i can also use lenght of heights_cdf
        distance.append(abs(heights_steps[i]-heights_cdf[i]))
    statistic=max(distance)
    return statistic
  
  
  
  
  #function to calculate p-value from sampler and statistic
  def ks_pvalue(statistic,distribution):
    #integration 
    x=[]
    for i in range (0,10000):
        if distribution[i]>=statistic:#data greater than the vertical line
            x.append(distribution[i])
    pvalue=len(x)/10000
    return pvalue 
