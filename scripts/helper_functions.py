#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:03:14 2025

@author: liebermanlab
"""

#functions used for Markey 2025, "low biomass project" skin microbiome analysis 


#%%set up environment
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import scipy
import colorcet as cc
#%%helper functions for Markey 2025

#making a helper function for filtering
#stuff that happens prior to the function
    #subset your dataset by extraction method if necessary
    #include both df with metadata and df that is only numeric data
    #define the species / genus that make up the input list

#this function for both 16S and shotgun metagenomic sequencing (all relative abundance inputs)
def set_threshold_rev(df_meta, df, input_taxa):
    m2=df_meta[df_meta["source"]=="mock2"].index
    m3=df_meta[df_meta["source"]=="mock3"].index
    m4=df_meta[df_meta["source"]=="mock4"].index
    numspecies=[]
    thresh=[]
    numspecies_input=[]
    data=df
    for n in np.arange(0,0.05, 0.0001):
        data[data<n]=0
        abovezero=data.ne(0).sum(axis=1)
        inputabovezero=data[input_taxa].ne(0).sum(axis=1)
        numspecies_input.append(inputabovezero)
        numspecies.append(abovezero)
        thresh.append(n)
    numspecies_df=pd.concat(numspecies, axis=1)
    inputdf=pd.concat(numspecies_input,axis=1)
    noninputdf=numspecies_df-inputdf
    inputdf.loc["thresh"]=thresh
    noninputdf.loc["thresh"]=thresh
    plotnoninput=noninputdf.T
    m3thresh=plotnoninput[plotnoninput[m3.values[0]]==0]["thresh"].min()
    #plot non input species and threshold lin
    plt.rcParams["figure.figsize"] = (5,3)
    fig1,ax = plt.subplots()
    sns.lineplot(data=plotnoninput, x="thresh", y=m2.values[0], color="orange")
    sns.lineplot(data=plotnoninput, x="thresh", y=m3.values[0], color="blue")
    sns.lineplot(data=plotnoninput, x="thresh", y=m4.values[0], color="green")
    plt.vlines(m3thresh, -0.1,10, color="purple", alpha=0.7)
    orange_l = mpatches.Patch(color='orange', label='mock 10^-2')
    blue_l = mpatches.Patch(color='blue', label='mock 10^-3')
    green_l = mpatches.Patch(color='green', label='mock 10^-4')
    plt.legend(handles=[orange_l, blue_l, green_l], bbox_to_anchor=(1.0,1.0))
    plt.xlim(-0.001,0.05)
    plt.ylim(-0.1,10)
    plt.xlabel("")
    plt.ylabel("")
    return noninputdf, fig1,m3thresh # this is the "set threshold" type graph

#this function for both 16S and shotgun metagenomic sequencing (all relative abundance inputs)
def perc_retain (leg,fore,buffer,air,m3t):
    leg_filt_perc_retain=[]
    fore_filt_perc_retain=[]
    air_filt_perc_retain=[]
    buffer_filt_perc_retain=[]
    thresh=[]
    for n in np.arange(0,0.05,0.0001):
        leg[leg<n]=0
        fore[fore<n]=0
        air[air<n]=0
        buffer[buffer<n]=0
        leg_filt_perc_retain.append(leg.sum(axis=1).div(1))
        fore_filt_perc_retain.append(fore.sum(axis=1).div(1))
        air_filt_perc_retain.append(air.sum(axis=1).div(1))
        buffer_filt_perc_retain.append(buffer.sum(axis=1).div(1))
        thresh.append(n)
    leg_retaindf=pd.concat(leg_filt_perc_retain, axis=1)
    leg_retaindf.loc["thresh"]=thresh
    fore_retaindf=pd.concat(fore_filt_perc_retain, axis=1)
    fore_retaindf.loc["thresh"]=thresh
    air_retaindf=pd.concat(air_filt_perc_retain,axis=1)
    air_retaindf.loc["thresh"]=thresh
    buff_retaindf=pd.concat(buffer_filt_perc_retain,axis=1)
    buff_retaindf.loc["thresh"]=thresh
    plotlegretain=leg_retaindf.T
    plotforeretain=fore_retaindf.T
    plotairretain=air_retaindf.T
    plotbuffretain=buff_retaindf.T
    meltleg=plotlegretain.melt(id_vars="thresh")
    meltfore=plotforeretain.melt(id_vars="thresh")
    meltair=plotairretain.melt(id_vars="thresh")
    meltbuff=plotbuffretain.melt(id_vars="thresh")
    #plot threshold against percent sample retained colored leg/fore head, blanks separate
    fig2,ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (5,3)
    sns.lineplot(data=meltair, x="thresh", y="value", color="#dddddd", errorbar="se")
    sns.lineplot(data=meltbuff, x="thresh", y="value", color="darkgrey", errorbar="se")
    sns.lineplot(data=meltleg, x="thresh", y="value", color="#b2849a", errorbar="se")
    sns.lineplot(data=meltfore, x="thresh", y="value", color="#911c1c", errorbar="se")
    plt.xlim(-0.001,0.051)
    plt.vlines(m3t, 0,1, color="purple")
    plt.ylim(0,1)
    plt.xlabel("")
    plt.ylabel("")
    #plt.ylabel("Fraction original sample retained")
    #plt.xlabel("Threshold ratio abundance")
    leg_l = mpatches.Patch(color='#b2849a', label='leg')
    fore_l = mpatches.Patch(color='#911c1c', label='forehead')
    buff_l = mpatches.Patch(color='darkgrey', label='extraction')
    air_l= mpatches.Patch(color="#dddddd", label="collection")
    plt.legend(handles=[air_l, buff_l, leg_l, fore_l], loc="lower right")
    return fig2,meltleg,meltfore

#need a different set of functions for qPCR because absolute abundance is on a different scale for choosing a threshold

def set_threshold_qpcr(df_meta,df, input_taxa):
    #get indexes for each dilution factor and use to subset for plotting different colors and averages
    m2=df_meta[df_meta["source"]=="mock2"].index
    m3=df_meta[df_meta["source"]=="mock3"].index
    m4=df_meta[df_meta["source"]=="mock4"].index
    numspecies=[]
    thresh=[]
    numspecies_input=[]
    data=df
    for n in np.arange(0,5000,100):
        data[data<n]=0
        abovezero=data.ne(0).sum(axis=1)
        inputabovezero=data[input_taxa].ne(0).sum(axis=1)
        numspecies.append(abovezero)
        numspecies_input.append(inputabovezero)
        thresh.append(n)
    numspecies_df=pd.concat(numspecies, axis=1)
    inputdf=pd.concat(numspecies_input, axis=1)
    noninputdf=numspecies_df-inputdf
    inputdf.loc["thresh"]=thresh
    noninputdf.loc["thresh"]=thresh
    #flip this to plot
    plotnoninput=noninputdf.T
    m3thresh=plotnoninput[plotnoninput[m3.values[0]]==0]["thresh"].min()
    #plot non input species and threshold line
    plt.rcParams["figure.figsize"] = (5,3)
    fig1,ax = plt.subplots()
    sns.lineplot(data=plotnoninput, x="thresh", y=m2.values[0], color="orange")
    sns.lineplot(data=plotnoninput, x="thresh", y=m3.values[0], color="blue")
    sns.lineplot(data=plotnoninput, x="thresh", y=m4.values[0], color="green")
    plt.vlines(m3thresh, -0.1,10, color="purple", alpha=0.7)
    orange_l = mpatches.Patch(color='orange', label='mock 10^-2')
    blue_l = mpatches.Patch(color='blue', label='mock 10^-3')
    green_l = mpatches.Patch(color='green', label='mock 10^-4')
    plt.legend(handles=[orange_l, blue_l, green_l], bbox_to_anchor=(1.0,1.0))
    #plt.text(m3thresh+10,8, str(np.round(m3thresh,4)))
    plt.xlim(-0.5,5015)
    plt.ylim(-0.1,10)
    plt.xlabel("")
    plt.ylabel("")
    #plt.title("Non-input species per sample per threshold")
    return fig1,m3thresh # this is the "set threshold" type graph

#this function for qPCR or other absolute abundance inputs
def perc_retain_qpcr (leg,fore,buffer,air,m3t):
    og_leg_total=leg.sum(axis=1)
    og_fore_total=fore.sum(axis=1)
    og_air_total=air.sum(axis=1)
    og_buffer_total=buffer.sum(axis=1)
    leg_filt_perc_retain=[]
    fore_filt_perc_retain=[]
    air_filt_perc_retain=[]
    buffer_filt_perc_retain=[]
    thresh=[]
    for n in np.arange(0,5000,100):
        leg[leg<n]=0
        fore[fore<n]=0
        air[air<n]=0
        buffer[buffer<n]=0
        leg_filt_perc_retain.append(leg.sum(axis=1).div(og_leg_total))
        fore_filt_perc_retain.append(fore.sum(axis=1).div(og_fore_total))
        air_filt_perc_retain.append(air.sum(axis=1).div(og_air_total))
        buffer_filt_perc_retain.append(buffer.sum(axis=1).div(og_buffer_total))
        thresh.append(n)
    leg_retaindf=pd.concat(leg_filt_perc_retain, axis=1)
    leg_retaindf.loc["thresh"]=thresh
    fore_retaindf=pd.concat(fore_filt_perc_retain, axis=1)
    fore_retaindf.loc["thresh"]=thresh
    air_retaindf=pd.concat(air_filt_perc_retain,axis=1)
    air_retaindf.loc["thresh"]=thresh
    buff_retaindf=pd.concat(buffer_filt_perc_retain,axis=1)
    buff_retaindf.loc["thresh"]=thresh
    plotlegretain=leg_retaindf.T
    plotforeretain=fore_retaindf.T
    plotairretain=air_retaindf.T
    plotbuffretain=buff_retaindf.T
    meltleg=plotlegretain.melt(id_vars="thresh")
    meltfore=plotforeretain.melt(id_vars="thresh")
    meltair=plotairretain.melt(id_vars="thresh")
    meltbuff=plotbuffretain.melt(id_vars="thresh")
    #plot threshold against percent sample retained colored leg/fore head, blanks separate
    fig2,ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (5,3)
    sns.lineplot(data=meltair, x="thresh", y="value", color="#dddddd")
    sns.lineplot(data=meltbuff, x="thresh", y="value", color="darkgrey")
    sns.lineplot(data=meltleg, x="thresh", y="value", color="#b2849a")
    sns.lineplot(data=meltfore, x="thresh", y="value", color="#911c1c")
    plt.xlim(-0.5,5015)
    plt.vlines(m3t, 0,1, color="purple")
    plt.ylim(0,1)
    plt.xlabel("")
    plt.ylabel("")
    #plt.ylabel("Fraction original sample retained")
    #plt.xlabel("Threshold ratio abundance")
    leg_l = mpatches.Patch(color='#b2849a', label='leg')
    fore_l = mpatches.Patch(color='#911c1c', label='forehead')
    buff_l = mpatches.Patch(color='darkgrey', label='extraction')
    air_l= mpatches.Patch(color="#dddddd", label="collection")
    plt.legend(handles=[air_l, buff_l, leg_l, fore_l], loc="lower right")
    print(meltleg[meltleg["thresh"]==m3t]["value"].median())
    print(meltfore[meltfore["thresh"]==m3t]["value"].median())
    return fig2


def viz_unfilt_16s (df_meta,palette):
    #split by site and convert to numeric
    leg_numeric=df_meta[df_meta["source"]=="leg"].iloc[:,:-16].apply(pd.to_numeric).reset_index(drop=True)
    fore_numeric=df_meta[df_meta["source"]=="forehead"].iloc[:,:-16].apply(pd.to_numeric).reset_index(drop=True)
    blank_numeric=df_meta[(df_meta["source"]=="air")|(df_meta["source"]=="buffer")].iloc[:,:-16].apply(pd.to_numeric).reset_index(drop=True)
    mock_numeric=df_meta[df_meta["sampletype"]=="lm_mock"].iloc[:,:-16].apply(pd.to_numeric)
    #get average
    fore_ave=pd.DataFrame(fore_numeric.mean(axis=0))
    leg_ave=pd.DataFrame(leg_numeric.mean(axis=0))
    mock_ave=pd.DataFrame(mock_numeric.mean(axis=0)).T
    #leg_fore_ave_df
    leg_fore_ave=pd.concat([leg_ave, fore_ave],axis=1)
    leg_fore_df=leg_fore_ave.T
    #add blank samples
    #get top 10 taxa in blanks
    blank_numeric.loc["ave"]=blank_numeric.mean(axis=0)
    topblank=pd.Series(blank_numeric.sort_values(by="ave", axis=1, ascending=False).columns[:20])
    mockinput=pd.Series(["Cutibacterium", "Staphylococcus", "Corynebacterium", "Escherichia"])
    #include both top 10 blank and mock inputs in barplot
    plotme_unfilt=pd.Series(pd.concat([mockinput,topblank]).unique())
    #make small and skinny
    #this is fig2a
    plt.rcParams["figure.figsize"] = (4,4)
    blank_aveskin=pd.concat([blank_numeric.iloc[:-1,:],leg_fore_df,mock_ave])
    blank_aveskin=blank_aveskin.fillna(0)
    blank_aveskin=blank_aveskin.loc[:, (blank_aveskin != 0).any(axis=0)]
    #add random unknown column to fill all samples to 1
    blank_aveskin["other"]=1-blank_aveskin[plotme_unfilt].sum(axis=1)
    #add other to list of columns to plot
    finalplot=pd.concat([plotme_unfilt, pd.Series("other")])
    print(blank_aveskin)
    barplot=blank_aveskin[finalplot].plot.bar(stacked=True, color=palette, width=0.9, linewidth=1, edgecolor="black")
    plt.xticks([0,1,2,3,4], ["Collection", "Extraction", "Leg", "Forehead","Mock"],rotation=15, fontsize=12)
    plt.ylim(0,1)
    plt.ylabel("")
    plt.legend(bbox_to_anchor=(1.0,1.0))
    return barplot


def viz_unfilt_qpcr (df_meta, palette):
    #split by site and convert to numeric
    leg_numeric=df_meta[df_meta["source"]=="leg"].iloc[:,2:-15].apply(pd.to_numeric)
    fore_numeric=df_meta[df_meta["source"]=="forehead"].iloc[:,2:-15].apply(pd.to_numeric)
    blank_numeric=df_meta[(df_meta["source"]=="air")|(df_meta["source"]=="buffer")].iloc[:,2:-15].apply(pd.to_numeric)
    mock_numeric=df_meta[df_meta["sampletype"]=="lm_mock"].iloc[:,2:-15].apply(pd.to_numeric)
    #get average
    #adding labels to make sure that they are in the right order and exist for plotting
    fore_ave=pd.DataFrame(fore_numeric.mean(axis=0)).T
    fore_ave["label"]=4
    leg_ave=pd.DataFrame(leg_numeric.mean(axis=0)).T
    leg_ave["label"]=3
    mock_ave=pd.DataFrame(mock_numeric.mean(axis=0)).T
    mock_ave["label"]=5
    #add blank samples
    #get top 20 taxa in blanks
    aveblank=blank_numeric.mean(axis=0)
    nonzero=pd.Series(aveblank[aveblank>0].sort_values(ascending=False).index[:20])
    mockinput=pd.Series(["Cutibacterium acnes", "Staphylococcus epidermidis", "Staphylococcus aureus","Escherichia coli"])
    #include both top 10 blank and mock inputs in barplot
    plotme_unfilt=pd.Series(pd.concat([mockinput,nonzero]).unique())
    #make small and skinny
    #this is fig2a
    plt.rcParams["figure.figsize"] = (4,4)
    blank_numeric["label"]=[1,2]
    blank_aveskin=pd.concat([blank_numeric,leg_ave,fore_ave,mock_ave])
    blank_aveskin=blank_aveskin.fillna(0)
    blank_aveskin=blank_aveskin.loc[:, (blank_aveskin != 0).any(axis=0)]
    #add random unknown column to fill all samples to 1
    blank_aveskin["other"]=blank_aveskin.iloc[:,:-1].sum(axis=1)-blank_aveskin[plotme_unfilt].sum(axis=1)
    #add other to list of columns to plot
    finalplot=pd.concat([plotme_unfilt, pd.Series("other")])
    #make sure you have the right number of bars in the expected order
    labelist=np.arange(1,6)
    blank_aveskin=blank_aveskin.merge(pd.DataFrame({"label":labelist}), on="label", how="right")
    barplot=blank_aveskin[finalplot].plot.bar(stacked=True, color=palette, width=0.9, linewidth=1, edgecolor="black")
    plt.xticks([0,1,2,3,4], ["Collection", "Extraction", "Leg", "Forehead","Mock"],rotation=15, fontsize=12)
    plt.ylim(10,10e6)
    #plt.ylabel("")
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.0,1.0))
    return barplot

def find_contam(df, sample_id):
    #calculate BC distance between all pairs of samples not from same subject #
    #provide numeric  df for bcdist and list of sample names
    #returns scatterplot of BC dist vs number of species and a list of possible splash samples
    pdist=scipy.spatial.distance.pdist(np.array(df),metric="braycurtis")
    square_pdist=pd.DataFrame(scipy.spatial.distance.squareform(pdist))
    square_pdist.columns=sample_id
    square_pdist.index=sample_id

    #reshape to remove lower triangle duplicate comparisons
    lower_t_nan = square_pdist.where(np.triu(np.ones(square_pdist.shape)).astype(np.bool))
    pdist_stack = pd.DataFrame(lower_t_nan.stack())
    pdist_stack.index=pdist_stack.index.set_names(["A", "B"])

    #only include non-matching samples; there are no duplicates because you removed lower triangle
    nomatch=pdist_stack.query('A != B').dropna()
    nomatch = nomatch.reset_index(level='B').astype("str")
    nomatch=nomatch.reset_index(level="A").astype("str")
    nomatch["samplepair"]=nomatch["A"]+str(" ")+nomatch["B"]
    allpairs_bcdist=nomatch.rename(columns={"A":"sample-id", "B":"compare_sample"})

    #get number of species per each sample 
    numspecies=pd.DataFrame({"num_species":df.ne(0).sum(axis=1), "sample-id":sample_id.astype('str')})
    allpairs_withspecies=allpairs_bcdist.merge(numspecies, how="inner", on="sample-id")
    allpairs_withspecies[[0,"num_species"]]=allpairs_withspecies[[0,"num_species"]].apply(pd.to_numeric)
    splash_check=allpairs_withspecies[(allpairs_withspecies["num_species"]>=allpairs_withspecies["num_species"].mean())&(allpairs_withspecies[0]<0.1)]["samplepair"]

    #plot number of species versus bray curtis distance to visualize 
    pal_samples = sns.color_palette(cc.glasbey, n_colors=len(sample_id))
    plt.rcParams["figure.figsize"] = (5,4)
    s=sns.scatterplot(data=allpairs_withspecies, x=0, y="num_species", color="gray")
    plt.legend(bbox_to_anchor=(1.0,1.0))
    plt.xlim(0,1)
    plt.hlines(allpairs_withspecies["num_species"].median(), 0,1, linestyle="dashed")
    plt.vlines(0.1,0,30, linestyles="dashed")
    #plt.ylim(0,30)
    plt.ylabel("# species in sample A")
    plt.xlabel("Bray Curtis Distance")
    plt.legend('',frameon=False)
    plt.show()
    return (s, splash_check)


## Helper functions from Evan Qu for PHLAME figures ##
class Frequencies():
    '''
    Holds clade frequency information from a given sample
    '''
    
    def __init__(self, path_to_frequencies_file):
        
        self.freqs = pd.read_csv(path_to_frequencies_file,
                                       index_col=0)



def read_phlame_frequencies_NEW(sample_names,
                              phlame_out_dir,
                             reference_genome):
    '''
    Read and consolidate frequency files from an output directory
    '''
    
    jt_frequencies = pd.DataFrame()
    for sample_name in sample_names:
        
        frequencies_file = f'{phlame_out_dir}/{sample_name}_ref_{reference_genome}_frequencies.csv'
        
        sample_freqs = Frequencies(frequencies_file)
        sample_freqs.freqs = sample_freqs.freqs['Relative abundance']
        sample_freqs.freqs.name = sample_name

        jt_frequencies = pd.concat((jt_frequencies,sample_freqs.freqs), axis=1)
        
    return jt_frequencies.T
