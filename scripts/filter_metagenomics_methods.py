#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 10:41:10 2025

@author: liebermanlab
"""

#filter metagenomic dataset using various methods
#comparing "all neg taxa" to decontam (performed in R) to single average abundance threshold
#visualizations to see impact of filtering:
    #average leg, forehead, mock sample
    #perc retained for leg, forehead
    #number of species diversity metric for leg, forehead
#read in raw bracken file
#read in decontam list of contaminant taxa

#%%set up environment
#import os
import pandas as pd
from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
from skbio.diversity import alpha_diversity
import seaborn as sns

#%%color palettes
source_pal=dict(air="#dddddd", buffer="darkgrey", leg="#b2849a", forehead="#911c1c")
#figure size
plt.rcParams["figure.figsize"] = (4,3)
#%%read in files

#define working directory for file locations #
work_dir="/Users/liebermanlab/MIT Dropbox/Laura Markey/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/upload_code_data"

#read in metadata
meta=pd.read_csv(work_dir+"/data/compare_filters_data/min_sg_meta.csv")

#read in unfiltered bracken dataframe
#only has sample-id column for metadata
#first column is bizarre sample name so removing it upon input
unfilt_sg=pd.read_csv(work_dir+"/data/compare_filters_data/unfilt_brack_sampleid.csv").iloc[:,1:]

#read in decontam output: these are locations of columns identified as contaminants by either decontam method
#the "combined" decontam method is actually even more conservative so instead are manually combining output of either method
#reading in outputs of decontam run using R
#output of decontam is the column indexes of contaminants based on the relabun_export file above after removing first column
freq=pd.read_csv(work_dir+"/data/compare_filters_data/freq_contam_column_index.csv")
prev=pd.read_csv(work_dir+"/data/compare_filters_data/prevalence_contam_column_index.csv")

###filtering using decontam ###
#combine methods to get largest list of contaminants
allindices=pd.concat([freq['x'],prev['x']]).drop_duplicates()

#slice unfiltered bracken data and sum contaminant columns
contam_columns=unfilt_sg.iloc[:,allindices].apply(pd.to_numeric)

#filter bracken df to remove contaminant columns and then sum remaining
decontam_relabun=unfilt_sg.drop(columns=contam_columns.columns)
decontam_relabun["total"]=decontam_relabun.iloc[:,:-1].apply(pd.to_numeric).sum(axis=1)


#what sample is the m3 mock?
m3sample=meta[meta["source_x"]=="mock3"]["sample-id"]

#how many non-input taxa are above zero in mock 10^-3 at start? 1165 above zero - 5 input species = 1160
noninput_m3=unfilt_sg[unfilt_sg["sample-id"]==129].iloc[:,:-2].apply(pd.to_numeric).ne(0).sum(axis=1)

#how many non-input taxa are above zero in mock 10^-3 after filtering?
total_sp_post_filter=decontam_relabun[decontam_relabun["sample-id"]=="129"].iloc[:,:-2].apply(pd.to_numeric).ne(0).sum(axis=1)

#### filtering using a single average taxa threshold across all samples #####

#choosing a single threshold for abundance
average_taxa_abundance=unfilt_sg.iloc[:,:-1].apply(pd.to_numeric).mean(axis=0)*100 #as percent not ratio
binwidth=0.01
plt.hist(average_taxa_abundance, bins=np.arange(0, max(average_taxa_abundance)+binwidth, binwidth), color="green")
plt.xlim(0,2)
plt.ylim(0,20)
plt.ylabel("counts")
plt.xlabel("Ave taxa % abundance")
plt.show()

#threshold arbitrarily set to 1%
thresh_1=average_taxa_abundance[average_taxa_abundance<1].index

#filter by this arbitrary threshold
thresh_relabun=unfilt_sg.drop(columns=thresh_1)
#how many taxa are above zero in mock 10^-3 after this filter?
total_sp_post_filter2=thresh_relabun[thresh_relabun["sample-id"]==129].iloc[:,:-1].apply(pd.to_numeric).ne(0).sum(axis=1)

#what percentage of the starting samples remains after this filter
thresh_relabun["total"]=thresh_relabun.iloc[:,:-1].apply(pd.to_numeric).sum(axis=1)

#### filtering by removing top 10 taxa in negative controls #####
negsamples=meta[(meta["source_x"]=="air")|(meta["source_x"]=="buffer")]["sample-id"]
removetaxa=unfilt_sg[unfilt_sg["sample-id"].isin(negsamples)].iloc[:,:-1].apply(pd.to_numeric).mean().sort_values(ascending=False)[:10].index

negfilt_sg=unfilt_sg.drop(columns=removetaxa)

#how many taxa are above zero in mock 10^-3 after this filter?
total_sp_post_filter3=negfilt_sg[negfilt_sg["sample-id"]==129].iloc[:,:-1].apply(pd.to_numeric).ne(0).sum(axis=1)

#what percentage of the starting samples remains after this filter
negfilt_sg["total"]=negfilt_sg.iloc[:,:-1].apply(pd.to_numeric).sum(axis=1)

#%%visualize percent retained from each dataset split by leg/forehead/blank sample
#optimal filtering method percent retained read in as csv
optimal_perc_retain=pd.read_csv(work_dir+"/data/compare_filters_data/sgmetagenomics_perc_retain.csv")
optimal=pd.DataFrame({"optimal_retain":optimal_perc_retain["perc_retain"]*100, "sample-id":optimal_perc_retain["sample-id"]})

#other filtering methods perc retain == total column 
#combine the total and sample-id columns as dataframes to merge and visualize as a stripplot 
negfilt_perc=pd.DataFrame({"negfilt_retain":negfilt_sg["total"]*100, "sample-id":negfilt_sg["sample-id"]})
thresh_perc=pd.DataFrame({"thresh_retain":thresh_relabun["total"]*100, "sample-id":thresh_relabun["sample-id"]})
decontam_perc=pd.DataFrame({"decontam_retain":decontam_relabun["total"]*100, "sample-id":decontam_relabun["sample-id"]})

#combine dataframes into a single dataframe for plotting
#list of perc retain dfs to merge
merge_retain=[optimal,negfilt_perc, thresh_perc, decontam_perc]
perc_retain_combined = reduce(lambda left, right: pd.merge(left, right, on='sample-id'), merge_retain)
#add metadata
perc_retain_meta=perc_retain_combined.merge(meta, how="inner", on="sample-id")

#visualize as stripplot and rename xtick columns
meltleg_perc_retain=perc_retain_meta[perc_retain_meta["source_x"]=="leg"].melt(id_vars="sample-id", value_vars=["optimal_retain", "negfilt_retain", "thresh_retain", "decontam_retain"])
df_median=meltleg_perc_retain.groupby('variable', sort=True)["value"].median()
df_median=df_median.reindex(["optimal_retain", "negfilt_retain", "thresh_retain", "decontam_retain"])
p=sns.stripplot(data=meltleg_perc_retain, x="variable",color="#b2849a", y="value",order=["optimal_retain", "negfilt_retain", "thresh_retain", "decontam_retain"],s=8, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["value"].items()]
#plt.title("Leg Samples")
#plt.ylabel("Percent retained after filter")
plt.ylabel("")
plt.xlabel("")
plt.ylim(0,101)
#plt.xticks([0,1,2,3], ["This paper", "Negative \ncontrol", "Universal \nthreshold", "Decontam"])
plt.xticks([])
plt.savefig(work_dir+"/supp_figures/FigS5d_filter_method_leg_perc_retain.png", format="png", bbox_inches="tight")
plt.show()

meltfore_perc_retain=perc_retain_meta[perc_retain_meta["source_x"]=="forehead"].melt(id_vars="sample-id", value_vars=["optimal_retain", "negfilt_retain", "thresh_retain", "decontam_retain"])
df_median=meltfore_perc_retain.groupby('variable', sort=True)["value"].median()
df_median=df_median.reindex(["optimal_retain", "negfilt_retain", "thresh_retain", "decontam_retain"])
p=sns.stripplot(data=meltfore_perc_retain, x="variable",color="#911c1c", y="value",order=["optimal_retain", "negfilt_retain", "thresh_retain", "decontam_retain"],s=8, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["value"].items()]
#plt.title("Forehead Samples")
#plt.ylabel("Percent retained after filter")
plt.xlabel("")
plt.ylim(0,101)
plt.xticks([])
plt.ylabel("")
#plt.xticks([0,1,2,3], ["This paper", "Negative \ncontrol", "Universal \nthreshold", "Decontam"])
plt.savefig(work_dir+"/supp_figures/FigS5d_filter_method_fore_perc_retain.png", format="png", bbox_inches="tight")
plt.show()

#combined leg and forehead on one plot
melt_all_perc_retain=perc_retain_meta[(perc_retain_meta["source_x"]=="leg")|(perc_retain_meta["source_x"]=="forehead")].melt(id_vars=["sample-id", "source_x"], value_vars=["optimal_retain", "negfilt_retain", "thresh_retain", "decontam_retain"])
df_median=melt_all_perc_retain.groupby(['source_x','variable'], sort=True)["value"].median()
#df_median=df_median.reindex(["optimal_retain", "negfilt_retain", "thresh_retain", "decontam_retain"])
ax=sns.stripplot(data=melt_all_perc_retain, x="variable", y="value",hue="source_x", palette=source_pal,s=8, linewidth=1, edgecolor="gray", dodge=True)
# Calculate and add median lines
# plot the mean line
sns.boxplot(showmeans=False,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': True},
            whiskerprops={'visible': False},
            zorder=10,
            x="variable",
            y="value",
            hue="source_x",
            data=melt_all_perc_retain,
            showfliers=False,
            showbox=False,
            showcaps=False,
            dodge=True)
#plt.title("Forehead Samples")
#plt.ylabel("Percent retained after filter")
plt.legend(title="Source", loc="lower right")
plt.xlabel("")
plt.ylim(0,101)
plt.xticks([])
plt.ylabel("")
#plt.xticks([0,1,2,3], ["This paper", "Negative \ncontrol", "Universal \nthreshold", "Decontam"])
plt.savefig(work_dir+"/supp_figures/FigS5d_filter_method_all_samples_perc_retain.svg", format="svg", bbox_inches="tight")
plt.show()

#%%renormalize and add metadata to filtered datasets
#decontam dataset
norm_decontam=decontam_relabun.iloc[:,:-2].apply(pd.to_numeric).div(decontam_relabun["total"], axis=0)
norm_decontam["sample-id"]=decontam_relabun["sample-id"]
norm_decontam_meta=norm_decontam.merge(meta, how="inner", on="sample-id")

#universal threshold
norm_threshold=thresh_relabun.iloc[:,:-2].apply(pd.to_numeric).div(thresh_relabun["total"], axis=0)
norm_threshold["sample-id"]=thresh_relabun["sample-id"]
norm_threshold_meta=norm_threshold.merge(meta, how="inner", on="sample-id")

#negative control filter
norm_negfilt=negfilt_sg.iloc[:,:-2].apply(pd.to_numeric).div(negfilt_sg["total"], axis=0)
norm_negfilt["sample-id"]=negfilt_sg["sample-id"]
norm_negfilt_meta=norm_negfilt.merge(meta, how="inner", on="sample-id")

##### reading in the filtered dataframe from the filter threshold set in main analysis script ###
filt_optimal=pd.read_csv(work_dir+"/data/compare_filters_data/filtered_sg_meta_powersoil.csv")
filt_optimal_meta=filt_optimal.merge(meta, how="inner", on="sample-id")

#%%visualize average sample composition after filtering
# i really don't want to color key renormalize for each set 
#instead visualizing the top 50 species and adding to a color key
#starting from the main text metagenomics color key species and adding as needed
hexspecieskey=pd.read_csv(work_dir+"/data/hex_color_keys/hex_species_blank_filter.csv")
speciespalette=dict(zip(hexspecieskey["taxa"], hexspecieskey["color"]))

#average composition of leg and forehead for each method, top50 species

#universal threshold
leg_thresh=norm_threshold_meta[norm_threshold_meta["source_x"]=="leg"].iloc[:,:-6].mean().sort_values(ascending=False).iloc[:50]
fore_thresh=norm_threshold_meta[norm_threshold_meta["source_x"]=="forehead"].iloc[:,:-6].mean().sort_values(ascending=False).iloc[:50]

#negative control filter
leg_negfilt=norm_negfilt_meta[norm_negfilt_meta["source_x"]=="leg"].iloc[:,:-6].mean().sort_values(ascending=False).iloc[:50]
fore_negfilt=norm_negfilt_meta[norm_negfilt_meta["source_x"]=="forehead"].iloc[:,:-6].mean().sort_values(ascending=False).iloc[:50]

#decontam 
leg_decontam=norm_decontam_meta[norm_decontam_meta["source_x"]=="leg"].iloc[:,:-6].mean().sort_values(ascending=False).iloc[:50]
fore_decontam=norm_decontam_meta[norm_decontam_meta["source_x"]=="forehead"].iloc[:,:-6].mean().sort_values(ascending=False).iloc[:50]

#optimal threshold per sample from paper
leg_opt=filt_optimal_meta[filt_optimal_meta["source_x"]=="leg"].iloc[:,:-6].mean().sort_values(ascending=False).iloc[:50]
fore_opt=filt_optimal_meta[filt_optimal_meta["source_x"]=="forehead"].iloc[:,:-6].mean().sort_values(ascending=False).iloc[:50]

#combine average values in dataframe
average_leg_filter_methods=pd.concat([leg_opt, leg_negfilt,leg_thresh,leg_decontam],axis=1).fillna(0).sort_values(by=0, ascending=False)
average_fore_filter_methods=pd.concat([fore_opt, fore_negfilt,fore_thresh,fore_decontam],axis=1).fillna(0).sort_values(by=0, ascending=False)
#flip so that species are columns
plot_aveleg=average_leg_filter_methods.T
plot_avefore=average_fore_filter_methods.T

#figure out height of gray bar to plot behind colors to add up to 1
plot_aveleg["other"]=1-plot_aveleg.sum(axis=1)
plot_avefore["other"]=1-plot_avefore.sum(axis=1)

#manually set order to plot to put skin species at the top and group together genera
speciesorder=pd.read_csv(work_dir+"/data/compare_filters_data/order_species_filter.csv")["taxa"]
legorder=speciesorder[speciesorder.isin(plot_aveleg.columns)]
foreorder=speciesorder[speciesorder.isin(plot_avefore.columns)]
#leg samples: plot top 50 species calculated individually for each filter method

plot_aveleg[legorder].plot.bar(stacked=True, color=speciespalette, width=0.9, linewidth=1, edgecolor="darkgray")
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xticks([0,1,2,3], ["This paper", "Negative \ncontrol", "Universal \nthreshold", "Decontam"], rotation=0)
plt.ylim(0,1)
#plt.savefig("filter_method_leg_average.svg", format="svg")
plt.savefig(work_dir+"/supp_figures/FigS5b_filter_method_leg_average.svg", format="svg", bbox_inches="tight")
plt.show()

#forehead samples: plot top 50 species calculated individually for each filter method
plot_avefore[foreorder].plot.bar(stacked=True, color=speciespalette, width=0.9, linewidth=1, edgecolor="darkgray")
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xticks([0,1,2,3], ["This paper", "Negative \ncontrol", "Universal \nthreshold", "Decontam"], rotation=0)
plt.ylim(0,1)
#plt.savefig("filter_method_fore_average.svg", format="svg")
plt.savefig(work_dir+"/supp_figures/FigS5a_filter_method_fore_average.svg", format="svg", bbox_inches="tight")
plt.show()

#%%Visualize mock community samples after filtering

#slice out mocks from each filtering method 
thispaper_mock=filt_optimal_meta[filt_optimal_meta["sampletype_x"]=="lm_mock"].iloc[:,:-6]
negcontrol_mock=norm_negfilt_meta[norm_negfilt_meta["sampletype_x"]=="lm_mock"].iloc[:,:-6]
threshold_mock=norm_threshold_meta[norm_threshold_meta["sampletype_x"]=="lm_mock"].iloc[:,:-6]
decontam_mock=norm_decontam_meta[norm_decontam_meta["sampletype_x"]=="lm_mock"].iloc[:,:-6]

#read in hypothetical composition of mocks csv
mock_hypo=pd.read_csv(work_dir+"/data/raw_data/mockinputs.csv")
mockhypo2=mock_hypo.iloc[:-2,1:]
#combine into a single df
#include mock hypothetical community 
allmethod_mock=pd.concat([mockhypo2, thispaper_mock,negcontrol_mock,threshold_mock, decontam_mock,]).fillna(0)

#only include columns with values above 1 perc or 0.01
coltotal=allmethod_mock.sum(axis=0)
includecol=coltotal[coltotal>0.01].index
allmethod_mock2=allmethod_mock[includecol]
allmethod_mock2["other"]=1-allmethod_mock2.sum(axis=1)

#add spacer to distinguish between methods
allmethod_mock2["spacer"]=[1,3,4,5,7,8,9,11,12,13,15,16,17]
spacelist=np.arange(1,18)
allmethod_mock2=allmethod_mock2.merge(pd.DataFrame({"spacer":spacelist}), on="spacer", how="right")

#plot figure
plt.rcParams["figure.figsize"] = (5,3.5)
#manually ordered columns to sort by genus and likely source
mockfilterorder=pd.read_csv(work_dir+"/data/compare_filters_data/order_col_mock_filters.csv", header=None)
ax=allmethod_mock2[mockfilterorder[0]].iloc[:,:-1].plot.bar(stacked=True, color=speciespalette, width=0.9, linewidth=1, edgecolor="gray")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.ylim(0,1)
plt.ylabel("")
plt.xticks([])
plt.savefig(work_dir+"/supp_figures/FigS5D_filter_method_mocks.svg", format="svg")
plt.show()

#%% counting number of species per sample using alpha div metric sobs
#calcuating alpha div for each dataset individually
opt_count_fore=alpha_diversity(metric="sobs", counts=filt_optimal_meta[filt_optimal_meta["source_x"]=="forehead"].iloc[:,:-6], ids=filt_optimal_meta[filt_optimal_meta["source_x"]=="forehead"]["sample-id"])
opt_count_leg=alpha_diversity(metric="sobs", counts=filt_optimal_meta[filt_optimal_meta["source_x"]=="leg"].iloc[:,:-6], ids=filt_optimal_meta[filt_optimal_meta["source_x"]=="leg"]["sample-id"])

neg_count_fore=alpha_diversity(metric="sobs", counts=norm_negfilt_meta[norm_negfilt_meta["source_x"]=="forehead"].iloc[:,:-6], ids=norm_negfilt_meta[norm_negfilt_meta["source_x"]=="forehead"]["sample-id"])
neg_count_leg=alpha_diversity(metric="sobs", counts=norm_negfilt_meta[norm_negfilt_meta["source_x"]=="leg"].iloc[:,:-6], ids=norm_negfilt_meta[norm_negfilt_meta["source_x"]=="leg"]["sample-id"])

thresh_count_fore=alpha_diversity(metric="sobs", counts=norm_threshold_meta[norm_threshold_meta["source_x"]=="forehead"].iloc[:,:-6], ids=norm_threshold_meta[norm_threshold_meta["source_x"]=="forehead"]["sample-id"])
thresh_count_leg=alpha_diversity(metric="sobs", counts=norm_threshold_meta[norm_threshold_meta["source_x"]=="leg"].iloc[:,:-6], ids=norm_threshold_meta[norm_threshold_meta["source_x"]=="leg"]["sample-id"])

decontam_count_fore=alpha_diversity(metric="sobs", counts=norm_decontam_meta[norm_decontam_meta["source_x"]=="forehead"].iloc[:,:-6], ids=norm_decontam_meta[norm_decontam_meta["source_x"]=="forehead"]["sample-id"])
decontam_count_leg=alpha_diversity(metric="sobs", counts=norm_decontam_meta[norm_decontam_meta["source_x"]=="leg"].iloc[:,:-6], ids=norm_decontam_meta[norm_decontam_meta["source_x"]=="leg"]["sample-id"])

#combine forehead or leg datasets
foreheadcounts=pd.concat([opt_count_fore,neg_count_fore,thresh_count_fore,decontam_count_fore],axis=1, keys=["optimal", "neg", "thresh", "decontam"])
legcounts=pd.concat([opt_count_leg,neg_count_leg,thresh_count_leg,decontam_count_leg],axis=1,keys=["optimal", "neg", "thresh", "decontam"])
foreheadcounts["sample-id"]=foreheadcounts.index
legcounts["sample-id"]=legcounts.index

#melt for ease of plotting 
#leg samples

meltleg_counts=legcounts.melt(id_vars="sample-id", value_vars=["optimal", "neg", "thresh", "decontam"])
df_median=meltleg_counts.groupby('variable', sort=True)["value"].median()
df_median=df_median.reindex(["optimal", "neg", "thresh", "decontam"])
p=sns.stripplot(data=meltleg_counts, x="variable",color="#b2849a", y="value",order=["optimal", "neg", "thresh", "decontam"],s=8, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["value"].items()]
#plt.title("Leg samples")
#plt.ylabel("# of species/sample")
plt.ylabel("")
plt.xlabel("")
plt.yscale("log")
plt.ylim(0.1,10000)
#plt.xticks([0,1,2,3], ["This paper", "Negative \ncontrol", "Universal \nthreshold", "Decontam"], rotation=0)
plt.xticks([])
plt.savefig(work_dir+"/supp_figures/FigS5c_filter_method_num_species_leg.svg", format="svg", bbox_inches="tight")
plt.show()

#forehead samples

meltfore_counts=foreheadcounts.melt(id_vars="sample-id", value_vars=["optimal", "neg", "thresh", "decontam"])
df_median=meltfore_counts.groupby('variable', sort=True)["value"].median()
df_median=df_median.reindex(["optimal", "neg", "thresh", "decontam"])
p=sns.stripplot(data=meltfore_counts, x="variable",color="#911c1c", y="value",order=["optimal", "neg", "thresh", "decontam"],s=8, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["value"].items()]
#plt.title("Forehead samples")
#plt.ylabel("# of species/sample")
plt.ylabel("")
plt.xlabel("")
plt.yscale("log")
plt.ylim(0.1,10000)
plt.xticks([])
#plt.xticks([0,1,2,3], ["This paper", "Negative \ncontrol", "Universal \nthreshold", "Decontam"], rotation=0)
plt.savefig(work_dir+"/supp_figures/FigS5c_filter_method_num_species_fore.svg", format="svg", bbox_inches="tight")
plt.show()

