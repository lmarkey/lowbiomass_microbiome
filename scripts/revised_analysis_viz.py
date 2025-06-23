#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 11:21:29 2025

@author: liebermanlab
"""

# generate figures and analysis for low biomass paper
# read in filtered datasets output from read_filter_datasets.py
# read in log file summary from bowtie2 used to calculate human reads
# also need to read in curated hex color key for species and genus level visualization 

#%%set up environment
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from functools import reduce
from scipy import stats
import colorcet as cc
import scipy
from skbio.diversity import alpha_diversity
from skbio.stats import distance
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
#import matplotlib.patches as mpatches
import helper_functions as hf
import matplotlib.lines as mlines
#%%read in files

#define working directory for file locations #
work_dir="/Users/liebermanlab/MIT Dropbox/Laura Markey/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/upload_code_data"


#remove samples that are not included in analysis (subject 23) or which appear to be contaminated by neighbor (leg 13)
#method used to determine which samples were likely splashover contaminants described in cell below
#visualization of contamination method uses "sg_metagenomics_allsamples"
#all downstream analysis and visualization uses the dfs which have dropped the bad samples

dropsample=[115,112, 23,44,69,90,123,151,162,163,168] #sample id from splashover ps leg 13, ps leg 10 because of G vag mismatch between forehead and leg and all samples from subj 23; also blanks from a weird redo of mock communities I didn't use - 162,163,168

###shotgun metagenomics ###

#only powersoil samples were sequenced with shotgun metagenomics; using data from those tagmented with standard 1:50 dilution
#relative abundance data, filtered and unfiltered
sg_metagenomics_allsamples=pd.read_csv(work_dir+"/data/output_read_filter_datasets/sg_filtered_bracken.csv")
sg_metagenomics=sg_metagenomics_allsamples[~sg_metagenomics_allsamples["sample-id"].isin(dropsample)]
sg_metagenomics_nofilt=pd.read_csv(work_dir+"/data/output_read_filter_datasets/unfilt_bracken_sg_meta.csv")

#sub species analysis performed by Evan using PHLAME and summary for visualization generated using strainlevel_meta.py
#cacnes_strain_meta=pd.read_csv("/Users/liebermanlab/MIT Dropbox/Laura Markey/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/2024_03_sample_processing/2024_09_nextseq/metagenomics_subset_analysis/cacnes_phylogroup_nodups.csv")

#read in list of samples to be dropped to make sure they don't show up in any of the figures
drop_sg_samples=pd.read_csv(work_dir+"/data/output_read_filter_datasets/dropsg_lowreads.csv")
#log file summary for human reads data
#txt files grep-ed from bowtie2 log files
human_total=pd.read_csv(work_dir+"/data/raw_data/bt2_human/total.txt", sep="\t", header=None)
human_alignonce=pd.read_csv(work_dir+"/data/raw_data/bt2_human/aligned1.txt", sep="\t",header=None)
human_alignmulti=pd.read_csv(work_dir+"/data/raw_data/bt2_human/alignedmulti.txt", sep="\t", header=None)

#bacteria from bracken for comparing to human reads
microreads=pd.read_csv(work_dir+"/data/output_read_filter_datasets/bracken_micro_reads.csv")
#### 16S sequencing ####
#relative abundnace data, filtered and unfiltered
#extraction method datasets were filtered separately then concatenated
# collapsed to the genus level
amplicon_relabun=pd.read_csv(work_dir+"/data/output_read_filter_datasets/amp_16s_genus_filtered.csv")
amplicon_relabun=amplicon_relabun[~amplicon_relabun["sample-id"].isin(dropsample)]
amplicon_unfilt=pd.read_csv(work_dir+"/data/output_read_filter_datasets/genus_16S_unfiltered.csv")
drop_16s_samples=pd.read_csv(work_dir+"/data/output_read_filter_datasets/drop16s_lowreads.csv")
### qPCR results: LM qPCR for universal 16S set threshold for inclusion and compare biomass; BioMe qPCR skin and vaginal microbiome panels ###

#LM performed qPCR on unmodified samples using universal V3 16S primers
#16S adjusted for input volume and universal qPCR
qpcr_16s_adjust=pd.read_csv(work_dir+"/data/raw_data/16s_lm_withau.csv")
#hex color species and genus color keys

#BioMe "PMP" qPCR: skin microbiome assay run on all three sets of extractions; vaginal microbiome assay just run on PowerSoil samples
#df containing all taxa skin + vaginal panel
qPCR_relabun=pd.read_csv(work_dir+"/data/output_read_filter_datasets/filt_qpcr_rel_abundance_all.csv")
qPCR_relabun=qPCR_relabun[~qPCR_relabun['sample-id'].isin(dropsample)]
qPCR_absabun_allsamples=pd.read_csv(work_dir+"/data/output_read_filter_datasets/filt_qpcr_absolute_abundace_all.csv")
qPCR_absabun=qPCR_absabun_allsamples[~qPCR_absabun_allsamples["sample-id"].isin(dropsample)]
#rename m luteus to remove ATCC label
qPCR_relabun=qPCR_relabun.rename(columns={"Micrococcus luteus ATCC 12698":"Micrococcus luteus"})
qPCR_absabun=qPCR_absabun.rename(columns={"Micrococcus luteus ATCC 12698":"Micrococcus luteus"})

#qPCR data from all three methods but only the skin panel (fgt panel run on ps only)
qPCR_skin_relabun_allsamples=pd.read_csv(work_dir+"/data/output_read_filter_datasets/filt_qpcr_rel_abundance_skin_panel.csv")
qPCR_skin_relabun=qPCR_skin_relabun_allsamples[~qPCR_skin_relabun_allsamples["sample-id"].isin(dropsample)]
qPCR_skin_absabun_allsamples=pd.read_csv(work_dir+"/data/output_read_filter_datasets/filt_qpcr_absolute_abundace_skin_panel.csv")
qPCR_skin_absabun=qPCR_skin_absabun_allsamples[~qPCR_skin_absabun_allsamples["sample-id"].isin(dropsample)]
qPCR_skin_relabun=qPCR_skin_relabun.rename(columns={"Micrococcus luteus ATCC 12698":"Micrococcus luteus"})
qPCR_skin_absabun=qPCR_skin_absabun.rename(columns={"Micrococcus luteus ATCC 12698":"Micrococcus luteus"})

#unfiltered qPCR skin panel to compare composition of blanks across methods
qPCR_skin_unfilt_abs=pd.read_csv(work_dir+"/data/raw_data/panel_qpcr/skin_panel_full_samples.csv") #
#will calculate rel abundance in figure section

##### metadata file across all samples ###

meta=pd.read_csv(work_dir+"/data/samplemetadata.csv")
meta["subject"]=meta["subject"].replace("none", 0)
meta["subject"]=meta["subject"].astype("int64")
#subjectmeta=pd.read_csv("/Users/liebermanlab/MIT Dropbox/Laura Markey/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/2024_03_sample_processing/update_subject_meta.csv")

# fake csv with hypothetical inputs for mock community
mock_hypo=pd.read_csv(work_dir+"/data/raw_data/mockinputs.csv")

#color palettes by source and kit
method_pal=dict(readylyse_dil="#6587a1", powersoil="#465e70", zymobiomics="#b2c3d0", powersoil_tubes="green" )#make a palette to label things by source
source_pal=dict(air="#dddddd", buffer="darkgrey", leg="#b2849a", forehead="#911c1c",mock1="#296595", mock2="#6490bc", mock3="#98b8d8", mock4="#c3d1e5")

#some additional genera have been added here to visualize blanks
genus_color=pd.read_csv(work_dir+"/data/hex_color_keys/genus_color_barplot.csv")
genus_pal=dict(zip(genus_color["taxa"], genus_color["color"]))

#used for main figure and supp figures for filtered metagenomics and blanks #
hexcolors_sg_pmp=pd.read_csv(work_dir+"/data/hex_color_keys/rev4_with_top_blanks_hex.csv")
taxa_pal_sg_pmp=dict(zip(hexcolors_sg_pmp["taxa"], hexcolors_sg_pmp["color"]))

#used for comparison of filtering methods has the most species and colors
hexspecieskey=pd.read_csv(work_dir+"/data/hex_color_keys/hex_species_blank_filter.csv")
speciespalette=dict(zip(hexspecieskey["taxa"], hexspecieskey["color"]))

#%%Identification of splash contamination using filtered datasets ###

## looking for samples from different subjects that are more similar than is likely to happen naturally ##

#for powersoil samples using metagenomics total
#add metadata 
sg_meta=sg_metagenomics_allsamples.merge(meta, how="inner", on="sample-id")
#subset out skin samples
ps_sg_skin=sg_meta[(sg_meta["source"]=="leg")|(sg_meta["source"]=="forehead")]
#provide function with numeric df as well as list of sample ids
ps_fig, ps_checklist=hf.find_contam(ps_sg_skin.iloc[:,1:-16], ps_sg_skin["sample-id"])
ps_fig.get_figure().savefig(work_dir+"/supp_figures/FigS9B_ps_identify_splash_samples.png", bbox_inches="tight")

# for readylyse and zymo samples using qPCR absolute abundance df

#add metadata
qpcr_meta=qPCR_skin_relabun_allsamples.merge(meta, how="inner", on="sample-id")

## other methods not included as supplemental figures but code is included incase others would like to visualize ##
### readylyse samples ###
qpcr_rl=qpcr_meta[qpcr_meta["method"]=="readylyse_dil"]
qpcr_rl_skin=qpcr_rl[(qpcr_rl["source"]=="leg")|(qpcr_rl["source"]=="forehead")]
rl_fig, rl_checklist=hf.find_contam(qpcr_rl_skin.iloc[:,:-16], qpcr_rl_skin["sample-id"])
#rl_fig.get_figure().savefig(work_dir+"/rl_identify_splash_samples.png", bbox_inches="tight")

### zymo samples ###
#zymo samples subset
qpcr_zy=qpcr_meta[qpcr_meta["method"]=="zymobiomics"]
qpcr_zy_skin=qpcr_zy[(qpcr_zy["source"]=="leg")|(qpcr_zy["source"]=="forehead")]
zy_fig, zy_checklist=hf.find_contam(qpcr_zy_skin.iloc[:,:-16], qpcr_zy_skin["sample-id"])
#zy_fig.get_figure().savefig("/Users/liebermanlab/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/2024_03_sample_processing/main_analysis_fig/zy_identify_splash_samples.png", bbox_inches="tight")

# manually visualized each pair identified by this method:
    #is the barplot extremely similar?
    #do the samples look very different from their counterparts extracted with other methods?
    #if answer is yes to both of the above, the lower biomass sample was dropped
    #this was only true for the PS pair (115 and 116); all others had good agreement between extraction methods

## make barplots for PS example to show that PS is similar but RL and ZY are not ##
# PS sample id corresponds to subject 13 leg and subject 14 leg #
# RL sample ids: 15, 16 
# ZY missing subject 13 leg and subject 14 leg is just Malassezia (genus)
#so just comparing PS to RL

#PS rel abundance top 10 species metagenomics
ps_check=sg_meta[sg_meta["sample-id"].isin([115,116])].iloc[:,:-16]
ps_check.loc["ave"]=ps_check.mean(axis=0)
ps_check=ps_check.sort_values(by="ave", ascending=False, axis=1)
ps_check["other"]=1-ps_check.iloc[:-1,:10].sum(axis=1)
ps_plot=pd.concat([ps_check.iloc[:-1,:10], ps_check["other"]], axis=1).iloc[:-1,:]
# ps_plot.plot.bar(stacked=True, color=speciespalette)
# plt.ylim(0,1)
# plt.legend(bbox_to_anchor=(1.0,1.0))
# plt.show()

# qPCR panel rel abundance for RL for these subjects #
rl_zy_viz_check=qpcr_meta[qpcr_meta["sample-id"].isin([15,16])].iloc[:,:-16]
rl_zy_viz_check.loc["ave"]=rl_zy_viz_check.mean(axis=0)
rl_zy_viz_check=rl_zy_viz_check.sort_values(by="ave", ascending=False, axis=1)
rl_zy_viz_check["other"]=1-rl_zy_viz_check.iloc[:-1,:10].sum(axis=1)
rl_zy_viz_check_plot=pd.concat([rl_zy_viz_check.iloc[:-1,:10], rl_zy_viz_check["other"]], axis=1).iloc[:-1,:]
# rl_zy_viz_check_plot.plot.bar(stacked=True, color=speciespalette)
# plt.ylim(0,1)
# plt.legend(bbox_to_anchor=(1.0,1.0))
# plt.show()

#should I just combine these on one set of axes #
compare_all_samples=pd.concat([ps_plot,rl_zy_viz_check_plot]).fillna(0)
compare_all_samples.loc["ave"]=compare_all_samples.mean(axis=0)
compare_all_samples=compare_all_samples.sort_values(by="ave", ascending=False, axis=1)
compare_all_samples['other']=compare_all_samples.pop('other')
compare_all_samples.iloc[:-1,:].plot.bar(stacked=True, color=speciespalette,width=0.9, linewidth=0.9, edgecolor="black")
plt.ylim(0,1)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xticks([])
plt.savefig(work_dir+"/supp_figures/FigS9C_ps_rl_compare_barplots_pair_splash_samples.svg", format="svg", bbox_inches="tight")
plt.show()

#%% Percent Human Reads analysis and visualization: summarizing human-aligned reads from bowtie2 log file "grep" outputs
#due to inane filenaming in samples.csv now have to split a column to find sample name and metadata
sample_split=human_total[0].str.split("_", expand=True)[1]
#pull out sample id
sample_id=sample_split.str[-3:]
#create condition and choice list to identify dilution of beads used based on location 
cond_list=[sample_split.str.contains("forehead"), sample_split.str.contains("leg"), sample_split.str.contains("mock"), sample_split.str.contains("buffer"), sample_split.str.contains("air")]
choice=[sample_split.str[8:10], sample_split.str[3:5], sample_split.str[5:7], sample_split.str[6:8], sample_split.str[3:5]]
choice2=["forehead", "leg", "mock", "buffer", "air"]
sample_dil=np.select(cond_list, choice, "None")
sample_source=np.select(cond_list, choice2, "other")

#make a dataframe of summary stats from bowtie2 with sample id and associated metadata
summary_stats=pd.DataFrame({"sample-id":sample_id, "source":sample_source, "tag_bead_dil":sample_dil, "totalreads":human_total[0].str.split(" ", expand=True)[2].astype("int64"), "human_1x":human_alignonce[1].str.split(" ", expand=True)[0].astype("int64"), "human_multi":human_alignmulti[1].str.split(" ", expand=True)[0].astype("int64")})

# #remove duplicates
nodup_humanstats=summary_stats.drop_duplicates(keep="first")

#remove 1:20 bead dilution samples
nodup_humanstats=nodup_humanstats[nodup_humanstats["tag_bead_dil"]=="50"]
#replace 097 with 97
nodup_humanstats["sample-id"]=nodup_humanstats["sample-id"].replace("097", "97").astype('int64')

nodup_humanstats["totalhuman"]=nodup_humanstats["human_1x"]+nodup_humanstats["human_multi"]
nodup_humanstats["perc_human"]=nodup_humanstats["totalhuman"].div(nodup_humanstats["totalreads"])*100

## Figure S7: percent human stripplot by source
sourceorder=["air", "buffer", "leg", "forehead"]
summary_stats_filt=nodup_humanstats[~nodup_humanstats["sample-id"].isin(dropsample)]
summary_stats_filt=summary_stats_filt[~summary_stats_filt["sample-id"].isin(drop_sg_samples["sample-id"])]
df_median=summary_stats_filt[(summary_stats_filt["tag_bead_dil"]=="50")&(summary_stats_filt["source"]!="mock")].groupby('source', sort=True)["perc_human"].median()
df_median=df_median.reindex(sourceorder)
p=sns.stripplot(data=summary_stats_filt[(summary_stats_filt["tag_bead_dil"]=="50")&(summary_stats_filt["source"]!="mock")], x="source", y="perc_human",hue="source", palette=source_pal, s=8, linewidth=1, edgecolor="gray", order=sourceorder)
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["perc_human"].items()]
plt.ylabel("% reads aligned human", fontsize=12)
#plt.xticks([0,1,2,3], ["Collection", "Extraction", "Leg", "Forehead"], fontsize=12)
plt.xticks([])
plt.ylim(0,100)
plt.ylabel("")
plt.xlabel("")
plt.savefig(work_dir+"/supp_figures/FigS7A_perc_reads_human_stripplot.png", format="png", bbox_inches="tight")
plt.show()

###### Supplemental Figure 2: sequencing dataset QC metrics#####

### Kraken percent classified QC ###

os.chdir(work_dir+"/data/raw_data/metagenomics_sequencing/kraken2_reports")
dir=os.getcwd()
#combine all kraken files listed in dfs into a single csv based on value of the taxonomy_id column
dfs=[]
for filename in os.listdir(dir):
    if filename.endswith("krakenRep.txt"):
        og=pd.read_table(filename, header=None).iloc[:10,:]
        taxa=og[5]
        sample=str(filename)
        sample_df=pd.DataFrame({"top10_taxa":taxa, str(sample):og[1]})
        dfs.append(sample_df)
        continue
    else:
        continue
#combine into one df
merged_kraken_rep=reduce(lambda left,right: pd.merge(left,right,on=['top10_taxa'],how='outer'),dfs).fillna('0')
#flip so that samples are rows and taxa are columns
kraken_flip=merged_kraken_rep.T
kraken_col=kraken_flip.iloc[0,:]
#subset out to rename columns
kraken_reads=kraken_flip.iloc[1:,:]
kraken_reads.columns=kraken_col
#make metadata columns
#grab sample name
sample_kraken=pd.Series(kraken_reads.index).str.split("_Human_filt_krakenRep.txt", expand=True)[0]
#pull out sample id
sample_id_k=sample_kraken.str[-3:]
#create condition and choice list to identify dilution of beads used based on location 
cond_list=[sample_kraken.str.contains("forehead"), sample_kraken.str.contains("leg"), sample_kraken.str.contains("mock"), sample_kraken.str.contains("buffer"), sample_kraken.str.contains("air")]
choice=[sample_kraken.str[8:10], sample_kraken.str[3:5], sample_kraken.str[5:7], sample_kraken.str[6:8], sample_kraken.str[3:5]]
choice2=["forehead", "leg", "mock", "buffer", "air"]
sample_dil=np.select(cond_list, choice, "None")
sample_source=np.select(cond_list, choice2, "other")

kraken_reads=kraken_reads.reset_index()
kraken_reads["sample-id"]=sample_id_k
kraken_reads["tag_bead_dil"]=sample_dil
kraken_reads["sample-id"]=kraken_reads["sample-id"].replace("097", "97")
#remove 1:20 bead dilution
kraken_reads=kraken_reads[kraken_reads["tag_bead_dil"]=="50"]

# just need sample-id and unclassified reads number
kraken_unclassified=kraken_reads[["sample-id", "unclassified"]].apply(pd.to_numeric)


#now kraken_reads contains the sample-id and the number of unclassified reads #
#merge with human nodups to get human reads and total reads
sg_stats_all=kraken_unclassified.merge(nodup_humanstats, how="inner", on="sample-id")
sg_stats_all["perc_unclassified"]=sg_stats_all["unclassified"].div(sg_stats_all["totalreads"])*100
#write to csv to format further for supplemental table
sg_stats_all.to_csv(work_dir+"/main_supp_tables/metagenomics_stats.csv")

#add other metadata
sg_stats_meta=sg_stats_all.merge(meta, how="inner", on="sample-id")
cond_list=[sg_stats_meta["source_y"]=="mock1",sg_stats_meta["source_y"]=="mock2", sg_stats_meta["source_y"]=="mock3", sg_stats_meta["source_y"]=="mock4", sg_stats_meta["source_y"]=="air", sg_stats_meta["source_y"]=="buffer"]
choice=["mock", "mock", "mock", "mock", "negative", "negative"]
sg_stats_meta["plotcat"]=np.select(cond_list, choice, sg_stats_meta["source_y"])


#plot perc unclassified metagenomics

plt.rcParams["figure.figsize"] = (4,4)
df_median=sg_stats_meta.groupby('plotcat', sort=True)["perc_unclassified"].median()
df_median=df_median.reindex(["negative","mock", "leg", "forehead"])
p=sns.stripplot(data=sg_stats_meta, x="plotcat", y="perc_unclassified", hue="source_y", palette=source_pal, order=["negative", "mock", "leg", "forehead"], s=8, edgecolor="gray", linewidth=1)
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["perc_unclassified"].items()]
plt.legend('',frameon=False)
plt.ylabel("")
plt.xlabel("")
plt.ylim(0,100)
plt.savefig(work_dir+"/supp_figures/FigS2C_kraken_perc_unclassified.png", format="png")
plt.show()

### poorly classified or undefined data from 16S ###
#read in file from read_filter_datasets
amplicon_qc_classification=pd.read_csv(work_dir+"/data/output_read_filter_datasets/16sqc_perc_undef_bad_class.csv")
#add metadata
amplicon_qc_class_meta=amplicon_qc_classification.merge(meta, how="inner", on="sample-id")
#subset just lowbiomass samples
plot_amp_qc=amplicon_qc_class_meta[amplicon_qc_class_meta["study"]=="lowbiomass"]
#making sure I did the math the same way
plot_amp_qc["perc_not_class"]=plot_amp_qc["bad_class_reads"].div(plot_amp_qc["totalreads"])*100

#subset out bonus buffers from later date
plot_amp_qc=plot_amp_qc[plot_amp_qc["date_processed"]!="20240711"]
plot_amp_qc=plot_amp_qc[plot_amp_qc["date_processed"]!="20240712"]
plot_amp_qc.to_csv(work_dir+"/main_supp_tables/amplicon_16s_sequencing_stats.csv")

#add new column for plotting 
cond_list=[plot_amp_qc["source"]=="mock1",plot_amp_qc["source"]=="mock2", plot_amp_qc["source"]=="mock3", plot_amp_qc["source"]=="mock4", plot_amp_qc["source"]=="air", plot_amp_qc["source"]=="buffer"]
choice=["mock", "mock", "mock", "mock", "negative", "negative"]
plot_amp_qc["plotcat"]=np.select(cond_list, choice, plot_amp_qc["source"])


plt.rcParams["figure.figsize"] = (4,4)
df_median=plot_amp_qc.groupby('plotcat', sort=True)["perc_not_class"].median()
df_median=df_median.reindex(["negative", "mock", "leg", "forehead"])
p=sns.stripplot(data=plot_amp_qc, x="plotcat", y="perc_not_class", hue="source", palette=source_pal, order=["negative", "mock", "leg", "forehead"], s=8, edgecolor="gray", linewidth=1)
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["perc_not_class"].items()]
plt.legend('',frameon=False)
plt.ylabel("")
plt.xlabel("")
plt.ylim(0,100)
plt.savefig(work_dir+"/supp_figures/FigS2D_16s_perc_undef_or_bact.png", format="png")
plt.show()

#%% Universal 16S qPCR performed by LM : compare sample biomass and set threshold for inclusion 

### Figure1:  plot stripplot awith AU normalized for extraction volume used###

#convert CT to AU using 4-fold mock dilution series of zymo samples
#did this in excel
#rl collected in 100 and used full volume for digestion then diluted 1:1 in water prior to qPCR - multiply by 2
#powersoil collected in 1000ul and used 250ul for sample extraction (and eluted in 50ul) - multiply by 4
#zymobiomics collected in 1000ul and used 400ul for sample extraction (and eluted in 50ul) - multiply by 2.5
sourceorder=["air", "buffer", "leg", "forehead", "mock1", "mock2", "mock3", "mock4"]
#eliminate hair samples by slicing on study
plotqpcr=qpcr_16s_adjust[qpcr_16s_adjust["study"]=="lowbio"]
plotqpcr=plotqpcr[~plotqpcr["sample-id"].isin(dropsample)]
df_median=plotqpcr.groupby('source', sort=True)["au_adjust"].median()
df_median=df_median.reindex(sourceorder)
plt.rcParams["figure.figsize"] = (5,4)
p=sns.stripplot(data=plotqpcr, x="source", y="au_adjust",hue="source", palette=source_pal, s=8, linewidth=1, edgecolor="gray", order=sourceorder)
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["au_adjust"].items()]
#plt.legend("")
plt.xticks([])
#plt.hlines(plotqpcr[plotqpcr["source"]=="mock4"]["au_adjust"].median(), -0.5,7.5, color="black", linestyle="dashed")
plt.yscale("log")
#plt.ylabel("16S qPCR (arbitrary units)")
plt.ylabel("")
p.spines['top'].set_visible(False)
p.spines['right'].set_visible(False)
plt.yticks(fontsize=12)
#plt.title("Universal 16S rRNA Signal")
plt.xlabel("")
plt.savefig(work_dir+"/main_figures/Fig1B_uni16s_adjust_allsamples.png", format="png", bbox_inches="tight")
plt.show()

leg_vs_air=stats.mannwhitneyu(plotqpcr[plotqpcr["source"]=="leg"]["au_adjust"], plotqpcr[(plotqpcr["source"]=="air")|(plotqpcr["source"]=="buffer")]["au_adjust"])


### Fig. 4A: biomass from each extraction method, split by location 
swabs=plotqpcr[(plotqpcr["source"]=="leg")|(plotqpcr["source"]=="forehead")] #slicing just skin swab samples
fig,axs=plt.subplots(1,2,sharey=True, figsize=(6,3))
df_med1=swabs[swabs["source"]=="forehead"].groupby("method",sort=True)["au_adjust"].median()
df_med1=df_med1.reindex(["powersoil", "zymobiomics", "readylyse_dil"])
p1=sns.stripplot(ax=axs[0], data=swabs[swabs["source"]=="forehead"], x="method", y="au_adjust", hue="method", palette=method_pal, order=["powersoil", "zymobiomics", "readylyse_dil"], linewidth=1, edgecolor="gray")
_ = [p1.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_med1.reset_index()["au_adjust"].items()]
axs[0].set_xticks([])
axs[0].set_ylabel("16S qPCR (arbitrary units)")
axs[0].set_yscale("log")
axs[0].set_xlabel("")
axs[0].spines[['top','right']].set_visible(False)
df_med2=swabs[swabs["source"]=="leg"].groupby("method",sort=True)["au_adjust"].median()
df_med2=df_med2.reindex(["powersoil", "zymobiomics", "readylyse_dil"])
p2=sns.stripplot(ax=axs[1], data=swabs[swabs["source"]=="leg"], x="method", y="au_adjust", hue="method", palette=method_pal, order=["powersoil", "zymobiomics", "readylyse_dil"], linewidth=1, edgecolor="gray")
_ = [p2.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_med2.reset_index()["au_adjust"].items()]
axs[1].set_xticks([])
axs[1].set_xlabel("")
axs[1].spines[['top', 'left','right']].set_visible(False)
# ps_l = mpatches.Patch(color='#465e70', label='PowerSoil')
# zy_l = mpatches.Patch(color='#b2c3d0', label='ZymoBIOMICS')
# rl_l = mpatches.Patch(color='#6587a1', label='ReadyLyse')
# plt.legend(handles=[ps_l, rl_l,zy_l], bbox_to_anchor=(1.0,1.0))
plt.savefig(work_dir+"/main_figures/Fig4A_swab_16s_dna_split_method.svg", format="svg")
plt.show()

#stats to compare biomass from each extraction method 
#stats to compare normalized 16S signal
#forehead
au_ps_zy_f=stats.mannwhitneyu(swabs[(swabs["source"]=="forehead")&(swabs["method"]=="powersoil")]["au_adjust"], swabs[(swabs["source"]=="forehead")&(swabs["method"]=="zymobiomics")]["au_adjust"])
au_ps_rl_f=stats.mannwhitneyu(swabs[(swabs["source"]=="forehead")&(swabs["method"]=="powersoil")]["au_adjust"], swabs[(swabs["source"]=="forehead")&(swabs["method"]=="readylyse_dil")]["au_adjust"])
au_zy_rl_f=stats.mannwhitneyu(swabs[(swabs["source"]=="forehead")&(swabs["method"]=="zymobiomics")]["au_adjust"], swabs[(swabs["source"]=="forehead")&(swabs["method"]=="readylyse_dil")]["au_adjust"])
#legs
au_ps_zy_l=stats.mannwhitneyu(swabs[(swabs["source"]=="leg")&(swabs["method"]=="powersoil")]["au_adjust"], swabs[(swabs["source"]=="leg")&(swabs["method"]=="zymobiomics")]["au_adjust"])
au_ps_rl_l=stats.mannwhitneyu(swabs[(swabs["source"]=="leg")&(swabs["method"]=="powersoil")]["au_adjust"], swabs[(swabs["source"]=="leg")&(swabs["method"]=="readylyse_dil")]["au_adjust"])
au_zy_rl_l=stats.mannwhitneyu(swabs[(swabs["source"]=="leg")&(swabs["method"]=="zymobiomics")]["au_adjust"], swabs[(swabs["source"]=="leg")&(swabs["method"]=="readylyse_dil")]["au_adjust"])

### Supplemental Figure 1B: collection order doesn't change biomass collected
fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (6,4)
methodlabels=["PowerSoil", "ReadyLyse", "ZymoBIOMICS"]
lm16sau_meta=plotqpcr.merge(meta[["sample-id", "quadrant"]], how="inner", on="sample-id")
df_median=lm16sau_meta.groupby('quadrant', sort=True)["au_adjust"].median()
df_median=df_median.reindex(["a", "b", "c", "d"])
p=sns.stripplot(data=lm16sau_meta[lm16sau_meta["quadrant"]!="none"], x="quadrant", y="au_adjust",hue="method", palette=method_pal, s=8, linewidth=1, edgecolor="gray", order=["a", "b", "c", "d"])
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["au_adjust"].items()]
#plt.legend("")
plt.xticks(rotation=0, fontsize=14)
#plt.hlines(lm16sau_meta[lm16sau_meta["source"]=="mock4"]["au_adjust"].mean(), -0.5,3.5, color="black", linestyle="dashed")
plt.yscale("log")
plt.ylabel("16S qPCR (arbitrary units)", fontsize=14)
handles, previous_labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=methodlabels, bbox_to_anchor=(1.0,1.0), title="Extraction Method")
#plt.title("Universal 16s rRNA Signal")
plt.xlabel("")
plt.savefig(work_dir+"/supp_figures/FigS1A_swab_dna_split_by_quadrant.png", format="png", bbox_inches="tight")
plt.show()

#%% calculating total number of reads included in metagenomics analysis and creating df for human-micro read ratio ##

##delete me if code runs ##

# #human reads are in nodup_humanstats
# #micro reads from bracken are microreads
# #16s data is in lm16sau_meta au_adjust

# #slice just the 1:50 bead dilution for nodup_humanstats
tagbead50=nodup_humanstats[nodup_humanstats["tag_bead_dil"]=="50"]

#combine all three to calculate ratio and plot against 16S data
human_micro_reads=tagbead50.merge(microreads, how="inner", on="sample-id")
# #when you merge human_micro_reads and the 16S data, samples get scrambled 
# #subset 16S data for powersoil samples - maybe this is why it's weird
ps_16s=lm16sau_meta[lm16sau_meta["method"]=="powersoil"]
#merge human reads and bracken microbe-assigned reads with 16s data subset
human_micro_reads_16s=human_micro_reads.merge(ps_16s,how="outer", on="sample-id")
human_micro_reads_16s["nonhumanreads"]=human_micro_reads_16s["totalreads"]-human_micro_reads_16s["totalhuman"]

#how many reads are included in analysis 
finalsgdata=human_micro_reads_16s[~human_micro_reads_16s["sample-id"].isin(drop_sg_samples["sample-id"])]
finalsgdata=finalsgdata[~finalsgdata["sample-id"].isin(dropsample)]
swabfinal=finalsgdata[(finalsgdata["source_x"]=="leg")|(finalsgdata["source_x"]=="forehead")]
negcontrols=finalsgdata[(finalsgdata["source_x"]=="buffer")|(finalsgdata["source_x"]=="air")]
swab_median_reads=swabfinal["totalreads"].median()

#nonhuman reads per sample type
nonhuman_med=finalsgdata.groupby("source_x")["nonhumanreads"].median()

#%% Supplemental Fig 7C: comparing human:micro read ratio to adjusted 16S values ###

#human reads are in nodup_humanstats
#micro reads from bracken are microreads
#16s data is in lm16sau_meta au_adjust

#remove bad samples
filt_human_micro_16s=human_micro_reads_16s[~human_micro_reads_16s["sample-id"].isin(dropsample)]
filt_human_micro_16s=filt_human_micro_16s[~filt_human_micro_16s["sample-id"].isin(drop_sg_samples["sample-id"])]

#add ratio of microbial to human reads
filt_human_micro_16s["ratio"]=filt_human_micro_16s["microbe_reads"].div(filt_human_micro_16s["totalhuman"])
#save the log10 of values as a new variable
filt_human_micro_16s["logratio"]=np.log(filt_human_micro_16s["ratio"])
filt_human_micro_16s["log_au"]=np.log(filt_human_micro_16s["au_adjust"])
filt_human_micro_16s=filt_human_micro_16s.replace(-np.inf, np.nan)

#split by source to plot
#forehead
fore=filt_human_micro_16s[filt_human_micro_16s["source_x"]=="forehead"]
sns.lmplot(data=fore, x="logratio", y="log_au", hue="source_x", palette=source_pal, ci=None)
plt.legend('',frameon=False)
plt.ylabel("")
plt.xlabel("")
plt.savefig(work_dir+"/supp_figures/FigS7C_forehead_human-micro_16s_scatterplot.svg", format="svg")
plt.show()
#leg
leg=filt_human_micro_16s[filt_human_micro_16s["source_x"]=="leg"]
sns.lmplot(data=leg, x="logratio", y="log_au", hue="source_x", palette=source_pal, ci=None)
plt.legend('',frameon=False)
plt.ylabel("")
plt.xlabel("")
plt.savefig(work_dir+"/supp_figures/FigS7C_leg_human-micro_16s_scatterplot.svg", format="svg")
plt.show()

#getting r and pvalue for that line
forehead_pearson=stats.pearsonr(fore["logratio"], fore["log_au"])
nonan_leg=leg.dropna()
leg_pearson=stats.pearsonr(nonan_leg["logratio"], nonan_leg["log_au"])
print(forehead_pearson)
print(leg_pearson)


#plot all samples together colored by source
allswabs=filt_human_micro_16s[(filt_human_micro_16s["source_x"]=="forehead")|(filt_human_micro_16s["source_x"]=="leg")]
sns.lmplot(data=allswabs, x="logratio", y="log_au", ci=None,  scatter_kws={'color': 'white'}, line_kws={'color':'gray'})
sns.scatterplot(data=allswabs, x="logratio", y="log_au", hue="source_x", palette=source_pal)
plt.legend('',frameon=False)
plt.ylabel("")
plt.xlabel("")
plt.savefig(work_dir+"/supp_figures/FigS7C_allswabs_human-micro_16s_scatterplot.png", format="png")
plt.show()

allswab_pearson=stats.pearsonr(allswabs["logratio"], allswabs["log_au"])
#%% ###Organizational notes ###

#Figure 2 is about filtering: these figures were generated from the "read_filter_datasets.py" script
#Figure 3 compares different DNA analysis methods. 
    #Therefore we need to remove bad samples and F. magna and renormalize; then collapse all to genus level
#%% ### process species level data from qPCR and metagenomics
    ##visualize species level data from shotgun metagenomics ##
    ##filter and renormalize species level data from qPCR ##

#species (and then genera) present at 10% or more in at least 1 sample are colored; all else are gray
# make a list of these species and genera, then save to csv to manually add hex color codes

## Finegoldia magna is really over-represented in the qPCR dataset so we removed F. magna from that dataset and the sequencing datasets (small fraction of whole but for fairness)

#qpcr relative abundance all the targets
nofm_full_qpcr=qPCR_relabun.drop(columns="Finegoldia magna")
nofm_full_qpcr["total"]=nofm_full_qpcr.iloc[:,:-1].sum(axis=1)
nofm_full_qpcr_renorm=nofm_full_qpcr.iloc[:,:-2].div(nofm_full_qpcr["total"], axis=0)
nofm_full_qpcr_renorm["sample-id"]=qPCR_relabun["sample-id"]

#qpcr relative abundance just the skin panel 
nofm_full_qpcr_skin=qPCR_skin_relabun.drop(columns="Finegoldia magna")
nofm_full_qpcr_skin["total"]=nofm_full_qpcr_skin.iloc[:,:-1].sum(axis=1)
nofm_full_qpcr_skin_renorm=nofm_full_qpcr_skin.iloc[:,:-2].div(nofm_full_qpcr_skin["total"], axis=0)
nofm_full_qpcr_skin_renorm["sample-id"]=qPCR_skin_relabun["sample-id"]

#qpcr absolute abundance - all the targets - no renorm needed
nofm_qPCR_absabun=qPCR_absabun.drop(columns="Finegoldia magna")
nofm_qPCR_skin_abs=qPCR_skin_absabun.drop(columns="Finegoldia magna")

#metagenomics
nofm_full_sg=sg_metagenomics.drop(columns="Finegoldia magna")
nofm_full_sg["total"]=nofm_full_sg.iloc[:,:-1].sum(axis=1)
nofm_full_sg_renorm=nofm_full_sg.iloc[:,:-2].div(nofm_full_sg["total"], axis=0)
nofm_full_sg_renorm["sample-id"]=sg_metagenomics["sample-id"]


#qpcr datasets
qpcr_rel_abun_nofm2=nofm_full_qpcr_renorm.fillna(0)

qpcr_abs_abun_nofm2=nofm_qPCR_absabun.fillna(0)

#sg metagenomics dataset
## remove sg metagenomics samples with <100k microbial reads ##
sg_nofm_samplefilt=nofm_full_sg_renorm[~nofm_full_sg_renorm["sample-id"].isin(drop_sg_samples["sample-id"])]

### collapse taxa for coloring purposes ###

#list of samples from each source
legsamplelist=meta[meta["source"]=="leg"]["sample-id"]
foresamplelist=meta[meta["source"]=="forehead"]["sample-id"]

##identify how many samples each taxa is present in above 10% in sg metagenomics ##
sg_above10perc_leg=[]
sg_above10perc_fore=[]
for c in sg_nofm_samplefilt.columns[:-1]:
    legs=sg_nofm_samplefilt[sg_nofm_samplefilt["sample-id"].isin(legsamplelist)]
    fore=sg_nofm_samplefilt[sg_nofm_samplefilt["sample-id"].isin(foresamplelist)]
    sg_above10perc_leg.append(len(legs[c][legs[c]>0.1]))
    sg_above10perc_fore.append(len(fore[c][fore[c]>0.1]))
sg_color_filt=pd.DataFrame({"taxa":sg_nofm_samplefilt.columns[:-1], "leg_10_perc":sg_above10perc_leg, "fore_10_perc":sg_above10perc_fore})

#create  list of species to color and genera to collapse
sg_species_color=sg_color_filt[(sg_color_filt["leg_10_perc"]>=1)|(sg_color_filt["fore_10_perc"]>=1)]["taxa"]

### collapse remaining species to the genus level and again list those at 10% or greater in at least 1 sample ###

## collapse sg metagenomics df to genus level for visualization ##

#collapse df
sg_collapse=pd.Series(sg_nofm_samplefilt.columns[~sg_nofm_samplefilt.columns.isin(sg_species_color)])
sg_collapse2=sg_collapse.str.split(" ", expand=True) #expand list of included species and split to see genus and species
sg_genus_list=pd.Series(sg_collapse2[0].unique())
sg_genus_list=sg_genus_list.replace("[Clostridium]", "Clostridium")
sg_genus_list=sg_genus_list.replace("[Ruminococcus]", "Ruminococcus")
gen_coll_data2=[]
for g in sg_genus_list[:-1]:
    columns_to_sum=sg_collapse[sg_collapse.str.contains(g)] #subset columns of same genera
    data=sg_nofm_samplefilt[columns_to_sum]
    genusdata=data.sum(axis=1)
    gen_coll_data2.append(genusdata)
sg_genus_coll_df=pd.concat(gen_coll_data2,axis=1)
sg_genus_coll_df.columns=sg_genus_list[:-1]

# subset such that genera present at 10% of 1 or more samples are colored and others are gray
sg_g_10perc=[]
for c in sg_genus_coll_df.columns:
    samples_10perc=len(sg_genus_coll_df[c][sg_genus_coll_df[c]>0.1])
    sg_g_10perc.append(samples_10perc)
sg_genus_color_filt=pd.DataFrame({"taxa":sg_genus_coll_df.columns, "num_above_5perc":sg_g_10perc})
sg_genus_color=sg_genus_color_filt[(sg_genus_color_filt["num_above_5perc"]>=1)&(sg_genus_color_filt["taxa"]!="sample-id")]["taxa"]
#manually add FGT genera to this list
fgtgenera=pd.Series(["Prevotella", "Aerococcus"])
#combine lists
combine_genera_color=pd.Series(pd.concat([sg_genus_color, fgtgenera]).unique())
#find gray genera
nocolor=sg_genus_list[~sg_genus_list.isin(combine_genera_color)]

#make a df and write to csv to manually add colors for species and genera
taxa_color_list=pd.DataFrame(pd.concat([sg_species_color,combine_genera_color,nocolor]))
taxa_color_list["color"]=np.where(taxa_color_list[0].isin(nocolor), "gray", "colorme")
taxa_color_list.to_csv(work_dir+"/data/hex_color_keys/combined_taxa_to_color_qpcr_sg.csv", index=None)

# I then manually added a bunch of species in order to visualize blanks and non-input species in mock
#so you need to reimport a csv that has these species added as well as the hex colors
hexcolors_sg_pmp=pd.read_csv(work_dir+"/data/hex_color_keys/rev4_with_top_blanks_hex.csv")
taxa_pal_sg_pmp=dict(zip(hexcolors_sg_pmp["taxa"], hexcolors_sg_pmp["color"]))

###combine species and genus level data###
#combine speces and genus level data for metagenomics
sg_coll_10perc=pd.concat([sg_genus_coll_df, sg_nofm_samplefilt[sg_species_color]],axis=1)
sg_coll_10perc["sample-id"]=sg_metagenomics["sample-id"]
sg_col_meta=sg_coll_10perc.merge(meta, how="inner", on="sample-id")
sg_col_meta["subject"]=sg_col_meta["subject"].replace("none", 0)
sg_col_meta["subject"]=sg_col_meta["subject"].astype("int64")

### datasets from above: ###
    #sample-filtered, taxa-filtered metagenomics and qPCR datasets: sg_nofm_samplefilt and qpcr_rel_abun_nofm2
    #filtered and collapsed at 10% abundance threshold datasets: sg_col_meta

#%%collapse metagenomics and qPCR to the genus level for comparison figures

##starting from the sample-filtered and taxa-filtered species level datasets##

#qPCR data collapse to genus level
pmpspecies=qpcr_rel_abun_nofm2 
pmpcollapse=pd.Series(pmpspecies.columns[:-2])
pmpcollapse2=pmpcollapse.str.split(pat=" ", expand=True).iloc[:,:2] #expand list of included species and split to see genus and species
pmp_genus_list=pd.Series(pmpcollapse2[0].unique())
gen_collapse_all=[]
for g in pmp_genus_list:
    columns_to_sum=pmpcollapse[pmpcollapse.str.contains(g)] #subset columns of same genera
    data=pmpspecies[columns_to_sum]
    genusdata=data.sum(axis=1)
    gen_collapse_all.append(genusdata)
pmpgenus_df=pd.concat(gen_collapse_all,axis=1)
pmpgenus_df.columns=pmp_genus_list
pmpgenus_df["sample-id"]=qpcr_rel_abun_nofm2["sample-id"]
pmpgenus_rel_meta=pmpgenus_df.merge(meta, how="inner", on="sample-id")

#collapse sg metagenomics data to genus level 
sg_species=sg_nofm_samplefilt
sgcollapse=pd.Series(sg_species.columns[:-1])
sgcollapse2=sgcollapse.str.split(pat=" ", expand=True).iloc[:,:2] #expand list of included species and split to see genus and species
sg_genus_list=pd.Series(sgcollapse2[0].unique()).replace("[Clostridium]","Clostridium")
sg_genus_list=sg_genus_list.replace("[Ruminococcus]", "Ruminococcus")
gen_collapse_all2=[]
for g in sg_genus_list:
    columns_to_sum=sgcollapse[sgcollapse.str.contains(g)] #subset columns of same genera
    data=sg_species[columns_to_sum]
    genusdata=data.sum(axis=1)
    gen_collapse_all2.append(genusdata)
sggenus_df=pd.concat(gen_collapse_all2,axis=1)
sggenus_df.columns=sg_genus_list
sggenus_df["sample-id"]=sg_nofm_samplefilt["sample-id"]
sggenus_rel_meta=sggenus_df.merge(meta, how="inner", on="sample-id") #44 genera

#%% remove bad samples and Finegoldia from 16S data

#removing samples based on number of reads assigned to taxa
amplicon_samplefilt=amplicon_relabun[~amplicon_relabun["sample-id"].isin(drop_16s_samples["sample-id"])]

#removing Finegoldia and renormalizing
amplicon_nof=amplicon_relabun.drop(columns="Finegoldia")
amplicon_nof["total"]=amplicon_nof.iloc[:,:-1].sum(axis=1)
amplicon_nof_renorm=amplicon_nof.iloc[:,:-2].div(amplicon_nof["total"],axis=0)
amplicon_nof_renorm["sample-id"]=amplicon_relabun["sample-id"].astype('int64')
amplicon_meta=amplicon_nof_renorm.merge(meta, how="inner", on="sample-id")
#%% Remove viral and fungal reads from qPCR and shotgun metagenomics data to compare fairly to 16S dataset

####remove fungus and virus-assigned data from qPCR and metagenomics data and renormalize

#comparing to 16S at genus level so you can start from genus-collapsed dataset

#qPCR known list of targets: only fungus is Malassezia
qPCR_nofungus=pmpgenus_df.drop(columns="Malassezia").fillna(0)
qPCR_nofungus["total"]=qPCR_nofungus.iloc[:,:-1].sum(axis=1)
qpcr_bact_renorm=qPCR_nofungus.iloc[:,:-2].div(qPCR_nofungus["total"],axis=0)
qpcr_bact_renorm["sample-id"]=pmpgenus_df["sample-id"]
qpcr_bact_meta=qpcr_bact_renorm.merge(meta, how="inner", on="sample-id")

#shotgun metagenomics has a lot of fungal and viral reads to remove and renormlize
#make lists of genera to remove
sg_fungus=sggenus_df.columns[(sggenus_df.columns.str.contains("Malassezia"))|(sggenus_df.columns.str.contains("Candida"))|(sggenus_df.columns.str.contains("Aspergillus"))]
sg_virus=sggenus_df.columns[sggenus_df.columns.str.contains("virus")]
sg_discard=pd.concat([pd.Series(sg_fungus),pd.Series(sg_virus)])
sg_bacteria=sggenus_df.drop(columns=sg_discard)
sg_bacteria["total"]=sg_bacteria.iloc[:,:-1].sum(axis=1)
sg_bact_renorm=sg_bacteria.iloc[:,:-2].div(sg_bacteria["total"], axis=0)
sg_bact_renorm["sample-id"]=sggenus_df["sample-id"]
sg_bact_meta=sg_bact_renorm.merge(meta, how="inner", on="sample-id")
#%%#make a list of genera and save it so you can make a color key 
generatocolor=pd.concat([pd.Series(sg_bact_renorm.columns[:-1]),pd.Series(qpcr_bact_renorm.columns[:-1]),pd.Series(amplicon_nof_renorm.columns[:-1])])
pd.Series(generatocolor.unique()).to_csv(work_dir+"/data/hex_color_keys/genera_to_color_corr_graph.csv")
#read in hex colors
#some additional genera have been added here to visualize blanks
genus_color=pd.read_csv(work_dir+"/data/hex_color_keys/genus_color_barplot.csv")
genus_pal=dict(zip(genus_color["taxa"], genus_color["color"]))

#%%Figure 3: compare DNA analysis methods barplot of mocks and ave leg/fore

# powersoil samples collapsed to the genus level, removed f magna and fungus and virus genera
# qpcr_bact_meta = qPCR dataset
# sg_bact_meta = shotgun dataset
# amplicon_meta = 16S dataset

###Figure 3A: powersoil pro mock community composition of filtered data by analysis method ###

# only showing powersoil kit (common to all methods)
#showing all taxa and combined on one set of axes
#plotting all taxa above zero

## subset powersoil mock community ##
#qpcr
mock_pmp=qpcr_bact_meta[qpcr_bact_meta["sampletype"]=="lm_mock"]
mock_pmp_ps=mock_pmp[mock_pmp["method"]=="powersoil"]
mock_pmp_ps_plot=mock_pmp_ps.loc[:, (mock_pmp_ps != 0).any(axis=0)].iloc[:,:-15]
#metagenomics
mock_sg=sg_bact_meta[sg_bact_meta["sampletype"]=="lm_mock"]
plotmocksg=mock_sg.fillna(0)
plotmocksg=plotmocksg.loc[:, (plotmocksg != 0).any(axis=0)].iloc[:,:-15]
#amplicon
mock_16s=amplicon_meta[amplicon_meta["sampletype"]=="lm_mock"]
mock16_ps=mock_16s[mock_16s["method"]=="powersoil"].fillna(0)
mock16_ps_plot=mock16_ps.loc[:, (mock16_ps != 0).any(axis=0)].iloc[:,:-15]

## combine across anlaysis ##
#add hypothetical input
mock_hypo_genus=mock_hypo.iloc[:,:6]
genus_columns=["sample", "Corynebacterium", "Staphylococcus", "Staphylococcus", "Escherichia", "Cutibacterium"]
mock_hypo_genus.columns=genus_columns
#sum staphylococcus
mock_hypo_genus["sum_staph"]=mock_hypo_genus["Staphylococcus"].sum(axis=1)
mockhypo2=mock_hypo_genus.drop(columns=["Staphylococcus"])
mockhypo2=mockhypo2.rename(columns={"sum_staph":"Staphylococcus"})
#drop extra hypothetical rows
mockhypo3=mockhypo2.iloc[:-2,1:]

# combine all dna analysis methods with the hypothetical composition based on plating #
allmethod_ps_mock=pd.concat([mockhypo3, mock_pmp_ps_plot, plotmocksg,mock16_ps_plot]).fillna(0)
#only include columns with values above 1 perc or 0.01
coltotal=allmethod_ps_mock.sum(axis=0)
includecol=coltotal[coltotal>0.01].index
allmethod_ps_mock2=allmethod_ps_mock[includecol]
#add metadata for peace of mind and spacing

### plotting all three methods ###
plt.rcParams["figure.figsize"] = (4,3)
allmethod_ps_mock2["analysis"]=["hypo", "qpcr", "qpcr", "qpcr", "sg", "sg", "sg", "amp", "amp", "amp"]
allmethod_ps_mock2["spacer"]=[1,3,4,5,7,8,9,11,12,13]
spacelist=np.arange(1,14)
allmethod_ps_mock2=allmethod_ps_mock2.merge(pd.DataFrame({"spacer":spacelist}), on="spacer", how="right")
#setting order to plot genera
genusorder=["Cutibacterium", "Staphylococcus", "Corynebacterium", "Escherichia", "Dermacoccus", "Enterocloster"]
ax=allmethod_ps_mock2[genusorder].plot.bar(stacked=True, color=genus_pal, width=0.9, linewidth=1, edgecolor="gray")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.ylim(0,1)
plt.xticks([])
plt.savefig(work_dir+"/main_figures/Fig3A_qpcr_sg_amp_mock_dil_ps.svg", format="svg")
plt.show()

### Figure 3B: average composition of leg samples

## subset and average leg samples extracted with powersoil ##

#metagenomics
sg_ps_leg=sg_bact_meta[sg_bact_meta["source"]=="leg"].iloc[:,:-16].mean() #shotgun metagenomics ave
#qpcr
qpcr_ps_leg=qpcr_bact_meta[(qpcr_bact_meta["source"]=="leg")&(qpcr_bact_meta["method"]=="powersoil")].iloc[:,:-16].mean()
#amplicon 16S
amp_ps_leg=amplicon_meta[(amplicon_meta["source"]=="leg")&(amplicon_meta["method"]=="powersoil")].iloc[:,:-16].mean()

#combine df
ave_leg=pd.concat([qpcr_ps_leg,sg_ps_leg,amp_ps_leg],axis=1)
plotaveleg=ave_leg.T
plotaveleg.loc["ave"]=plotaveleg.mean(axis=0)
#sort by average abundance
plotaveleg=plotaveleg.sort_values(by="ave", axis=1, ascending=False).iloc[:-1,:]
#only plot taxa above 0
plotaveleg2=plotaveleg.loc[:, (plotaveleg != 0).any(axis=0)]

#subset out color columns for the front of list
popmegray=genus_color[genus_color["color"]=="gray"]["taxa"]
colorcol=pd.Series(plotaveleg2.columns[:-16][~plotaveleg2.columns[:-16].isin(popmegray)])
graycol=pd.Series(plotaveleg2.columns[plotaveleg2.columns.isin(popmegray)])
neworder=pd.concat([colorcol,graycol])

#plot data
plt.rcParams["figure.figsize"] = (3,3)
ax=plotaveleg2[neworder].plot.bar(stacked=True, color=genus_pal, width=0.9, linewidth=0.9, edgecolor="black")
plt.xticks([])
#plt.xticks([0,1,2], ["qPCR","Metagenomics", "16s amplicon"], rotation=0)
#plt.title("Average Leg Swab Composition")
plt.ylim(0,1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.savefig(work_dir+"/main_figures/Fig3B_average_leg_powersoil_by_analysis.svg", format="svg")
plt.show()

### Supplemental Figure 6A: average composition of forehead sampples

## subset and average forehad samples extracted with powersoil 
#metagenomics
sg_ps_fore=sg_bact_meta[sg_bact_meta["source"]=="forehead"].iloc[:,:-16].mean() #shotgun metagenomics ave
#qpcr
qpcr_ps_fore=qpcr_bact_meta[(qpcr_bact_meta["source"]=="forehead")&(qpcr_bact_meta["method"]=="powersoil")].iloc[:,:-16].mean()
#amplicon 16S
amp_ps_fore=amplicon_meta[(amplicon_meta["source"]=="forehead")&(amplicon_meta["method"]=="powersoil")].iloc[:,:-16].mean()

#combine df
ave_fore=pd.concat([qpcr_ps_fore,sg_ps_fore,amp_ps_fore],axis=1)
plotavefore=ave_fore.T
plotavefore.loc["ave"]=plotavefore.mean(axis=0)
#sort by average abundance
plotavefore=plotavefore.sort_values(by="ave", axis=1, ascending=False).iloc[:-1,:]
#only plot taxa above 0
plotavefore2=plotavefore.loc[:, (plotavefore != 0).any(axis=0)]

#subset out color columns for the front of list
colorcol=pd.Series(plotavefore2.columns[:-16][~plotavefore2.columns[:-16].isin(popmegray)])
graycol=pd.Series(plotavefore2.columns[plotavefore2.columns.isin(popmegray)])
neworder=pd.concat([colorcol,graycol])

#plot data
plt.rcParams["figure.figsize"] = (3,3)
plotavefore2[neworder].plot.bar(stacked=True, color=genus_pal, width=0.9, linewidth=0.9, edgecolor="black")
plt.xticks([])
#plt.xticks([0,1,2], ["qPCR","Metagenomics", "16s amplicon"], rotation=0)
#plt.title("Average Forehead Swab Composition")
plt.ylim(0,1)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.savefig(work_dir+"/supp_figures/FigS6A_average_forehead_powersoil_by_analysis.svg", format="svg")
plt.show()
#%%### figure 3 compare DNA analysis methods: Alpha Diversity: Fig. 3D and Supp Fig6B-C ###

## including all bacterial genera detected by any method but no fungus/virus ##
# powersoil samples collapsed to the genus level, removed f magna and fungus and virus genera
# qpcr_bact_meta = qPCR dataset with metadata
# sg_bact_meta = shotgun dataset with metadata
# amplicon_meta = 16S dataset with metadata

#metagenomics
sg_simpson=alpha_diversity(metric="simpson", counts=sg_bact_meta.iloc[:,:-16], ids=sg_bact_meta["sample-id"])
sg_counts=alpha_diversity(metric="sobs", counts=sg_bact_meta.iloc[:,:-16], ids=sg_bact_meta["sample-id"])
sg_alpha=pd.concat([sg_simpson,sg_counts],axis=1,keys=["sg_simpson", "sg_counts"])
sg_alpha["sample-id"]=sg_alpha.index
sg_alpha=sg_alpha.reset_index(drop=True)
#amplicon
amp_simpson=alpha_diversity(metric="simpson", counts=amplicon_meta.iloc[:,:-16], ids=amplicon_meta["sample-id"])
amp_counts=alpha_diversity(metric="sobs", counts=amplicon_meta.iloc[:,:-16], ids=amplicon_meta["sample-id"])
amp_alpha=pd.concat([amp_simpson,amp_counts],axis=1,keys=["amp_simpson", "amp_counts"])
amp_alpha["sample-id"]=amp_alpha.index
amp_alpha=amp_alpha.reset_index(drop=True)
#qpcr
qpcr_simpson=alpha_diversity(metric="simpson", counts=qpcr_bact_meta.iloc[:,:-16], ids=qpcr_bact_meta["sample-id"])
qpcr_counts=alpha_diversity(metric="sobs", counts=qpcr_bact_meta.iloc[:,:-16], ids=qpcr_bact_meta["sample-id"])
qpcr_alpha=pd.concat([qpcr_simpson,qpcr_counts],axis=1,keys=["qpcr_simpson", "qpcr_counts"])
qpcr_alpha["sample-id"]=qpcr_alpha.index
qpcr_alpha=qpcr_alpha.reset_index(drop=True)
#combine into one dataframe
merge1=sg_alpha.merge(amp_alpha, how="inner", on="sample-id")
alphamerge=merge1.merge(qpcr_alpha, how="inner", on="sample-id")
#add metadata
alpha_div_meta=alphamerge.merge(meta, how="inner", on="sample-id")

#melt for ease of plotting 
melt_simpson=alpha_div_meta.melt(id_vars=["subject", "source", "sampletype", "method"], value_vars=["sg_simpson", "amp_simpson", "qpcr_simpson"])
#split by source of swab

  ### Figure Supp 6B ###
plt.rcParams["figure.figsize"] = (5,4)
melt_simpson["subject"]=melt_simpson["subject"].astype("str")
fore_simpson=melt_simpson[melt_simpson["source"]=="forehead"]
df_median=fore_simpson.groupby('variable', sort=True)["value"].median()
df_median=df_median.reindex(["qpcr_simpson", "sg_simpson", "amp_simpson"])
p=sns.stripplot(data=fore_simpson, x="variable",color="#aba2c6", y="value",order=["qpcr_simpson", "sg_simpson", "amp_simpson"],s=8, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["value"].items()]
plt.legend('',frameon=False)
#plt.xticks([0,1,2], ["qPCR", "Metagenomics", "16s amplicon"])
plt.xticks([])
plt.xlabel("")
#plt.title("Alpha diversity Foreheads")
plt.ylabel("")
#plt.ylabel("Simpson diversity")
plt.ylim(0,1)
plt.savefig(work_dir+"/supp_figures/FigS6B_forehead_simpson.png", format="png")
plt.show()
mw_pcr_sg3=stats.mannwhitneyu(fore_simpson[fore_simpson["variable"]=="qpcr_simpson"]["value"], fore_simpson[fore_simpson["variable"]=="sg_simpson"]["value"])
mw_pcr_amp3=stats.mannwhitneyu(fore_simpson[fore_simpson["variable"]=="qpcr_simpson"]["value"], fore_simpson[fore_simpson["variable"]=="amp_simpson"]["value"])
mw_sg_amp3=stats.mannwhitneyu(fore_simpson[fore_simpson["variable"]=="sg_simpson"]["value"], fore_simpson[fore_simpson["variable"]=="amp_simpson"]["value"])


  ### Fig. 3c ### 
  
leg_simpson=melt_simpson[melt_simpson["source"]=="leg"]
df_median=leg_simpson.groupby('variable', sort=True)["value"].median()
df_median=df_median.reindex(["qpcr_simpson", "sg_simpson", "amp_simpson"])
p=sns.stripplot(data=leg_simpson, x="variable", y="value",order=["qpcr_simpson", "sg_simpson", "amp_simpson"],s=8,color="#aba2c6", linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["value"].items()]
plt.legend('',frameon=False)
plt.xticks([])
#plt.xticks([0,1,2], ["qPCR", "Metagenomics", "16s amplicon"])
plt.xlabel("")
#plt.title("Alpha diversity Legs")
plt.ylabel("Simpson diversity")
plt.ylabel("")
p.spines['top'].set_visible(False)
p.spines['right'].set_visible(False)
plt.ylim(0,1)
plt.savefig(work_dir+"/main_figures/Fig3C_leg_simpson.png", format="png")
plt.show()
#stats for leg simpson
mw_pcr_sg=stats.mannwhitneyu(leg_simpson[leg_simpson["variable"]=="qpcr_simpson"]["value"], leg_simpson[leg_simpson["variable"]=="sg_simpson"]["value"])
mw_pcr_amp=stats.mannwhitneyu(leg_simpson[leg_simpson["variable"]=="qpcr_simpson"]["value"], leg_simpson[leg_simpson["variable"]=="amp_simpson"]["value"])
mw_sg_amp=stats.mannwhitneyu(leg_simpson[leg_simpson["variable"]=="sg_simpson"]["value"], leg_simpson[leg_simpson["variable"]=="amp_simpson"]["value"])

  ### Fig. Supp 7B ###
#comparing sample diversity to % human reads
alphadiv_reads=alpha_div_meta.merge(nodup_humanstats, how="inner", on="sample-id")
#drop bad samples
alphadiv_reads=alphadiv_reads[~alphadiv_reads["sample-id"].isin(drop_sg_samples["sample-id"])]
sns.scatterplot(data=alphadiv_reads[alphadiv_reads["sampletype"]=="swab"], x="perc_human", y="sg_simpson", hue="source_x", palette=source_pal)
plt.ylim(-0.01,1)
plt.xlim(0,100)
plt.legend(title="Source")
#plt.xlabel("% human reads")
#plt.ylabel("Simpson diversity")
plt.xlabel("")
plt.ylabel("")
plt.savefig(work_dir+"/supp_figures/FigS7B_simpson_vs_perc_human.png", format="png")
plt.show()

#%%Fig 3: Beta Diversity comparing dna anlysis methods: Fig 3C

### Combine filtered and renomalized genus data into a single df from all three DNA analysis methods ###

#need to add "analysis" label to distinguish datasets
#these are just data + sample-id columns df

amplicon_nof_renorm["analysis"]="amp"
sg_bact_renorm["analysis"]="sg"
qpcr_bact_renorm["analysis"]="qpcr"

#combine all three anlayses into one giant df
sg_qpcr_amp_genus_bact=pd.concat([sg_bact_renorm,qpcr_bact_renorm,amplicon_nof_renorm])

#add metadata
combined_genus_bact_meta=sg_qpcr_amp_genus_bact.merge(meta, how="inner", on="sample-id")

#subset out just powersoil samples
combined_powersoil=combined_genus_bact_meta[combined_genus_bact_meta["method"]=="powersoil"]

#just compare distance between samples based on genera detected by all three methods
allthree=set(qpcr_bact_renorm.columns[:-2])&set(sg_bact_renorm.columns[:-2])&set(amplicon_nof_renorm.columns[:-2])
#remove sample
comb_share_taxa=combined_powersoil[list(allthree)]
comb_share_taxa["sample-id"]=combined_powersoil["sample-id"]
comb_share_taxa["analysis"]=combined_powersoil["analysis"]
comb_share_taxa_meta=comb_share_taxa.merge(meta, how="inner", on="sample-id")

#### compare for each subject+location sample the composition BC distance ###
## for foreheads you can do this in a straightforward way because none of the powersoil samples were dropped from analysis
#foreheads
sg_vs_pcr=[]
sgpcr_s=[]
sg_vs_amp=[]
sgamp_s=[]
amp_vs_pcr=[]
pcramp_s=[]
data=comb_share_taxa_meta[comb_share_taxa_meta["source"]=="forehead"].fillna(0)
for s in data["subject"].unique():
    sgdata=np.array(data[(data["subject"]==s)&(data["analysis"]=="sg")].iloc[:,:-17].apply(pd.to_numeric))
    ampdata=np.array(data[(data["subject"]==s)&(data["analysis"]=="amp")].iloc[:,:-17].apply(pd.to_numeric))
    pcrdata=np.array(data[(data["subject"]==s)&(data["analysis"]=="qpcr")].iloc[:,:-17].apply(pd.to_numeric))
    sg_pcr_dist=scipy.spatial.distance.cdist(sgdata, pcrdata,metric="braycurtis")
    sg_vs_pcr.append(sg_pcr_dist[0,0])
    sgpcr_s.append(s)
    sg_amp_dist=scipy.spatial.distance.cdist(sgdata, ampdata,metric="braycurtis")
    sg_vs_amp.append(sg_amp_dist[0,0])
    sgamp_s.append(s)
    pcr_amp_dist=scipy.spatial.distance.cdist(pcrdata, ampdata,metric="braycurtis")
    amp_vs_pcr.append(pcr_amp_dist[0,0])
    pcramp_s.append(s)
bcdist_fore_withinsub_commontaxa=pd.DataFrame({"subject":sgamp_s, "pcr_vs_sg":sg_vs_pcr, "pcr_vs_amp":amp_vs_pcr, "amp_vs_sg":sg_vs_amp})
fmw_qpcrsg_vs_qpcr16s=stats.mannwhitneyu(bcdist_fore_withinsub_commontaxa["pcr_vs_sg"], bcdist_fore_withinsub_commontaxa["pcr_vs_amp"])
fmw_qpcrsg_vs_sg16s=stats.mannwhitneyu(bcdist_fore_withinsub_commontaxa["pcr_vs_sg"], bcdist_fore_withinsub_commontaxa["amp_vs_sg"])
fmw_qpcr16s_vs_sg16s=stats.mannwhitneyu(bcdist_fore_withinsub_commontaxa["pcr_vs_amp"], bcdist_fore_withinsub_commontaxa["amp_vs_sg"])

##plotting forehead B-C Distance comparing analysis methods ##
# Fig. S6 #
plt.rcParams["figure.figsize"] = (5,4)
#as stripplots with median
#reshape data for plotting 
meltbc_fore=bcdist_fore_withinsub_commontaxa.melt(id_vars="subject")
meltbc_fore["subject"]=meltbc_fore["subject"].astype("str")
df_median=meltbc_fore.groupby('variable', sort=True)["value"].median()
df_median=df_median.reindex(["pcr_vs_sg","amp_vs_sg", "pcr_vs_amp"])
p=sns.stripplot(data=meltbc_fore, x="variable", y="value",color="#aba2c6",order=["pcr_vs_sg", "amp_vs_sg", "pcr_vs_amp"], s=8, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["value"].items()]
plt.legend('',frameon=False)
plt.ylabel("")
plt.xlabel("")
plt.xticks([])
#plt.xticks([0,1,2], ["qPCR vs shotgun","qPCR vs 16S", "16S vs shotgun" ])
#plt.title("Leg samples")
plt.ylim(0,1)
plt.savefig(work_dir+"/supp_figures/FigS6C_bcdist_fore_samples_commongenera.png", format="png")
plt.show()


#For leg samples this is tricky because a lot of leg samples were removed by universal 16S filter so you need to check to make sure they are non-zero for each subject
#combine separately and then merge in order to get around some subjects included in one assay but not another
#legs
sg_vs_amp=[]
sgamp_s=[]
data=comb_share_taxa_meta[comb_share_taxa_meta["source"]=="leg"].fillna(0)
#amplicon vs sg
for s in data["subject"].unique():
    sgdata=np.array(data[(data["subject"]==s)&(data["analysis"]=="sg")].iloc[:,:-17].apply(pd.to_numeric))
    ampdata=np.array(data[(data["subject"]==s)&(data["analysis"]=="amp")].iloc[:,:-17].apply(pd.to_numeric))
    if len(sgdata)!=0 and len(ampdata)!=0:
        sg_amp_dist=scipy.spatial.distance.cdist(sgdata, ampdata,metric="braycurtis")
        sg_vs_amp.append(sg_amp_dist[0,0])
        sgamp_s.append(s)
        continue
sg_amp_df=pd.DataFrame({"subject":sgamp_s, "sg_vs_amp":sg_vs_amp})

#sg vs pcr
sg_vs_pcr=[]
sgpcr_s=[]
for s in data["subject"].unique():
    sgdata=np.array(data[(data["subject"]==s)&(data["analysis"]=="sg")].iloc[:,:-17].apply(pd.to_numeric))
    pcrdata=np.array(data[(data["subject"]==s)&(data["analysis"]=="qpcr")].iloc[:,:-17].apply(pd.to_numeric))
    if len(sgdata)!=0 and len(pcrdata)!=0:
        sg_pcr_dist=scipy.spatial.distance.cdist(sgdata, pcrdata,metric="braycurtis")
        sg_vs_pcr.append(sg_pcr_dist[0,0])
        sgpcr_s.append(s)
        continue
sg_pcr_df=pd.DataFrame({"subject":sgpcr_s, "sg_vs_pcr":sg_vs_pcr})
#pcr vs amp
amp_vs_pcr=[]
pcramp_s=[]
for s in data["subject"].unique():
    ampdata=np.array(data[(data["subject"]==s)&(data["analysis"]=="amp")].iloc[:,:-17].apply(pd.to_numeric))
    pcrdata=np.array(data[(data["subject"]==s)&(data["analysis"]=="qpcr")].iloc[:,:-17].apply(pd.to_numeric))
    if len(ampdata)!=0 and len(pcrdata)!=0:
        amp_pcr_dist=scipy.spatial.distance.cdist(ampdata, pcrdata,metric="braycurtis")
        amp_vs_pcr.append(amp_pcr_dist[0,0])
        pcramp_s.append(s)
        continue
amp_pcr_df=pd.DataFrame({"subject":pcramp_s, "amp_vs_pcr":amp_vs_pcr})

merge1=sg_amp_df.merge(sg_pcr_df, how="outer", on="subject")
bcdist_leg_withinsub_filt_common_taxa=merge1.merge(amp_pcr_df, how="outer", on="subject")

#plot bc dist comparing analysis methods in pairwise fashion

  ### Figure 3D ###
plt.rcParams["figure.figsize"] = (5,4)
#as stripplots with median
#reshape data for plotting 
pal_subject=sns.color_palette(cc.glasbey, n_colors=21)
meltbc_leg=bcdist_leg_withinsub_filt_common_taxa.melt(id_vars="subject")
meltbc_leg["subject"]=meltbc_leg["subject"].astype("str")
df_median=meltbc_leg.groupby('variable', sort=True)["value"].median()
df_median=df_median.reindex(["sg_vs_pcr","sg_vs_amp", "amp_vs_pcr"])
p=sns.stripplot(data=meltbc_leg, x="variable", y="value",color="#aba2c6",order=["sg_vs_pcr", "sg_vs_amp", "amp_vs_pcr"], s=8, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["value"].items()]
plt.legend('',frameon=False)
plt.ylabel("")
plt.xlabel("")
plt.xticks([])
p.spines['top'].set_visible(False)
p.spines['right'].set_visible(False)
#plt.xticks([0,1,2], ["qPCR vs shotgun","qPCR vs 16S", "16S vs shotgun" ])
#plt.title("Leg samples")
plt.ylim(0,1.1)
plt.savefig(work_dir+"/main_figures/Fig3D_bcdist_leg_samples_commongenera.png", format="png")
plt.show()
#Mann whitney test for statistics shown on figure
mw_qpcrsg_vs_qpcr16s=stats.mannwhitneyu(bcdist_leg_withinsub_filt_common_taxa["sg_vs_pcr"].dropna(), bcdist_leg_withinsub_filt_common_taxa["amp_vs_pcr"].dropna())
mw_qpcrsg_vs_sg16s=stats.mannwhitneyu(bcdist_leg_withinsub_filt_common_taxa["sg_vs_pcr"].dropna(), bcdist_leg_withinsub_filt_common_taxa["sg_vs_amp"].dropna())
mw_qpcr16s_vs_sg16s=stats.mannwhitneyu(bcdist_leg_withinsub_filt_common_taxa["amp_vs_pcr"].dropna(), bcdist_leg_withinsub_filt_common_taxa["sg_vs_amp"].dropna())

#%%Fig 4: Beta Diversity comparing dna extraction methods: Fig 4

#For forehead and leg samples this is tricky because samples were removed by read filter from some but not all methods
#so you need to check to make sure they are non-zero for each subject
#combine separately and then merge in order to get around some subjects included in one assay but not another


## 16S data has more samples included but very limited resolution for legs ##

#foreheads
data=amplicon_meta[amplicon_meta["source"]=="forehead"].fillna(0)
#ps vs rl
ps_vs_rl=[]
subject_list=[]
for s in data["subject"].unique():
    psdata=np.array(data[(data["subject"]==s)&(data["method"]=="powersoil")].iloc[:,:-16].apply(pd.to_numeric))
    rldata=np.array(data[(data["subject"]==s)&(data["method"]=="readylyse_dil")].iloc[:,:-16].apply(pd.to_numeric))
    if len(psdata)!=0 and len(rldata)!=0:
        ps_rl_dist=scipy.spatial.distance.cdist(psdata, rldata,metric="braycurtis")
        ps_vs_rl.append(ps_rl_dist[0,0])
        subject_list.append(s)
        continue
ps_rl_fore_df=pd.DataFrame({"subject":subject_list, "ps_vs_rl":ps_vs_rl})

#ps vs zy
ps_vs_zy=[]
subject_list=[]
for s in data["subject"].unique():
    psdata=np.array(data[(data["subject"]==s)&(data["method"]=="powersoil")].iloc[:,:-16].apply(pd.to_numeric))
    zydata=np.array(data[(data["subject"]==s)&(data["method"]=="zymobiomics")].iloc[:,:-16].apply(pd.to_numeric))
    if len(psdata)!=0 and len(zydata)!=0:
        ps_zy_dist=scipy.spatial.distance.cdist(psdata, zydata,metric="braycurtis")
        ps_vs_zy.append(ps_zy_dist[0,0])
        subject_list.append(s)
        continue
ps_zy_fore_df=pd.DataFrame({"subject":subject_list, "ps_vs_zy":ps_vs_zy})

#rl vs zy
rl_vs_zy=[]
subject_list=[]
for s in data["subject"].unique():
    rldata=np.array(data[(data["subject"]==s)&(data["method"]=="readylyse_dil")].iloc[:,:-16].apply(pd.to_numeric))
    zydata=np.array(data[(data["subject"]==s)&(data["method"]=="zymobiomics")].iloc[:,:-16].apply(pd.to_numeric))
    if len(rldata)!=0 and len(zydata)!=0:
        rl_zy_dist=scipy.spatial.distance.cdist(rldata, zydata,metric="braycurtis")
        rl_vs_zy.append(rl_zy_dist[0,0])
        subject_list.append(s)
        continue
rl_zy_fore_df=pd.DataFrame({"subject":subject_list, "rl_vs_zy":rl_vs_zy})


#combine different comparisons with sequential merge
merge1=ps_rl_fore_df.merge(ps_zy_fore_df, how="outer", on="subject")
bcdist_fore_withinsub_methods=merge1.merge(rl_zy_fore_df, how="outer", on="subject")

#legs
data=amplicon_meta[amplicon_meta["source"]=="leg"].fillna(0)
#ps vs rl
ps_vs_rl=[]
subject_list=[]
for s in data["subject"].unique():
    psdata=np.array(data[(data["subject"]==s)&(data["method"]=="powersoil")].iloc[:,:-16].apply(pd.to_numeric))
    rldata=np.array(data[(data["subject"]==s)&(data["method"]=="readylyse_dil")].iloc[:,:-16].apply(pd.to_numeric))
    if len(psdata)!=0 and len(rldata)!=0:
        ps_rl_dist=scipy.spatial.distance.cdist(psdata, rldata,metric="braycurtis")
        ps_vs_rl.append(ps_rl_dist[0,0])
        subject_list.append(s)
        continue
ps_rl_leg_df=pd.DataFrame({"subject":subject_list, "ps_vs_rl":ps_vs_rl})

#ps vs zy
ps_vs_zy=[]
subject_list=[]
for s in data["subject"].unique():
    psdata=np.array(data[(data["subject"]==s)&(data["method"]=="powersoil")].iloc[:,:-16].apply(pd.to_numeric))
    zydata=np.array(data[(data["subject"]==s)&(data["method"]=="zymobiomics")].iloc[:,:-16].apply(pd.to_numeric))
    if len(psdata)!=0 and len(zydata)!=0:
        ps_zy_dist=scipy.spatial.distance.cdist(psdata, zydata,metric="braycurtis")
        ps_vs_zy.append(ps_zy_dist[0,0])
        subject_list.append(s)
        continue
ps_zy_leg_df=pd.DataFrame({"subject":subject_list, "ps_vs_zy":ps_vs_zy})

#rl vs zy
rl_vs_zy=[]
subject_list=[]
for s in data["subject"].unique():
    rldata=np.array(data[(data["subject"]==s)&(data["method"]=="readylyse_dil")].iloc[:,:-16].apply(pd.to_numeric))
    zydata=np.array(data[(data["subject"]==s)&(data["method"]=="zymobiomics")].iloc[:,:-16].apply(pd.to_numeric))
    if len(rldata)!=0 and len(zydata)!=0:
        rl_zy_dist=scipy.spatial.distance.cdist(rldata, zydata,metric="braycurtis")
        rl_vs_zy.append(rl_zy_dist[0,0])
        subject_list.append(s)
        continue
rl_zy_leg_df=pd.DataFrame({"subject":subject_list, "rl_vs_zy":rl_vs_zy})


#combine different comparisons with sequential merge
merge1=ps_rl_leg_df.merge(ps_zy_leg_df, how="outer", on="subject")
bcdist_leg_withinsub_methods=merge1.merge(rl_zy_leg_df, how="outer", on="subject")

#plot bc dist comparing extraction methods in pairwise fashion

#melt for easier viz
melt_fore_method_bc=bcdist_fore_withinsub_methods.melt(id_vars="subject")
melt_leg_method_bc=bcdist_leg_withinsub_methods.melt(id_vars="subject")

#split by source but show on same y axis

fig,axs=plt.subplots(1,2,sharey=True, figsize=(6,3))
#plot 1 forehead
df_med1=melt_fore_method_bc.groupby("variable",sort=True)["value"].median()
df_med1=df_med1.reindex(["ps_vs_zy", "ps_vs_rl", "rl_vs_zy"])
p1=sns.stripplot(ax=axs[0], data=melt_fore_method_bc, x="variable", y="value", color="#aba2c6", order=["ps_vs_zy", "ps_vs_rl", "rl_vs_zy"], linewidth=1, edgecolor="gray")
_ = [p1.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_med1.reset_index()["value"].items()]
axs[0].set_xticks([])
axs[0].set_ylabel("")
axs[0].set_xlabel("")
axs[0].spines[['top','right']].set_visible(False)
#plot 2 leg
df_med2=melt_leg_method_bc.groupby("variable",sort=True)["value"].median()
df_med2=df_med2.reindex(["ps_vs_zy", "ps_vs_rl", "rl_vs_zy"])
p2=sns.stripplot(ax=axs[1], data=melt_leg_method_bc, x="variable", y="value", color="#aba2c6", order=["ps_vs_zy", "ps_vs_rl", "rl_vs_zy"], linewidth=1, edgecolor="gray")
_ = [p2.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_med2.reset_index()["value"].items()]
axs[1].set_xticks([])
axs[1].set_xlabel("")
axs[1].spines[['top', 'left','right']].set_visible(False)
plt.savefig(work_dir+"/supp_figures/FigS4F_bc_dist_methods_16S.svg", format="svg")
plt.show()

#Mann whitney test for statistics shown on figure
mw_ps_zy_vs_ps_rl_fore=stats.mannwhitneyu(bcdist_fore_withinsub_methods["ps_vs_zy"].dropna(), bcdist_fore_withinsub_methods["ps_vs_rl"].dropna())
mw_ps_zy_vs_rl_zy_fore=stats.mannwhitneyu(bcdist_fore_withinsub_methods["ps_vs_zy"].dropna(), bcdist_fore_withinsub_methods["rl_vs_zy"].dropna())
mw_ps_rl_vs_rl_zy_fore=stats.mannwhitneyu(bcdist_fore_withinsub_methods["ps_vs_rl"].dropna(), bcdist_fore_withinsub_methods["rl_vs_zy"].dropna())

mw_ps_zy_vs_ps_rl_leg=stats.mannwhitneyu(bcdist_leg_withinsub_methods["ps_vs_zy"].dropna(), bcdist_leg_withinsub_methods["ps_vs_rl"].dropna())
mw_ps_zy_vs_rl_zy_leg=stats.mannwhitneyu(bcdist_leg_withinsub_methods["ps_vs_zy"].dropna(), bcdist_leg_withinsub_methods["rl_vs_zy"].dropna())
mw_ps_rl_vs_rl_zy_leg=stats.mannwhitneyu(bcdist_leg_withinsub_methods["ps_vs_rl"].dropna(), bcdist_leg_withinsub_methods["rl_vs_zy"].dropna())

### qPCR results for same comparisons - note that ZYMO leg samples are hot trash so leaving them out and just have ps_rl distance for legs ###
#dataset nofm_full_qpcr_skin_renorm

#add metadata to qpcr
qpcr_skin_meta=nofm_full_qpcr_skin_renorm.merge(meta, how="inner", on="sample-id")

#foreheads
data=qpcr_skin_meta[qpcr_skin_meta["source"]=="forehead"].fillna(0)
#ps vs rl
ps_vs_rl=[]
subject_list=[]
for s in data["subject"].unique():
    psdata=np.array(data[(data["subject"]==s)&(data["method"]=="powersoil")].iloc[:,:-16].apply(pd.to_numeric))
    rldata=np.array(data[(data["subject"]==s)&(data["method"]=="readylyse_dil")].iloc[:,:-16].apply(pd.to_numeric))
    if len(psdata)!=0 and len(rldata)!=0:
        ps_rl_dist=scipy.spatial.distance.cdist(psdata, rldata,metric="braycurtis")
        ps_vs_rl.append(ps_rl_dist[0,0])
        subject_list.append(s)
        continue
ps_rl_fore_df=pd.DataFrame({"subject":subject_list, "ps_vs_rl":ps_vs_rl})

#ps vs zy
ps_vs_zy=[]
subject_list=[]
for s in data["subject"].unique():
    psdata=np.array(data[(data["subject"]==s)&(data["method"]=="powersoil")].iloc[:,:-16].apply(pd.to_numeric))
    zydata=np.array(data[(data["subject"]==s)&(data["method"]=="zymobiomics")].iloc[:,:-16].apply(pd.to_numeric))
    if len(psdata)!=0 and len(zydata)!=0:
        ps_zy_dist=scipy.spatial.distance.cdist(psdata, zydata,metric="braycurtis")
        ps_vs_zy.append(ps_zy_dist[0,0])
        subject_list.append(s)
        continue
ps_zy_fore_df=pd.DataFrame({"subject":subject_list, "ps_vs_zy":ps_vs_zy})

#rl vs zy
rl_vs_zy=[]
subject_list=[]
for s in data["subject"].unique():
    rldata=np.array(data[(data["subject"]==s)&(data["method"]=="readylyse_dil")].iloc[:,:-16].apply(pd.to_numeric))
    zydata=np.array(data[(data["subject"]==s)&(data["method"]=="zymobiomics")].iloc[:,:-16].apply(pd.to_numeric))
    if len(rldata)!=0 and len(zydata)!=0:
        rl_zy_dist=scipy.spatial.distance.cdist(rldata, zydata,metric="braycurtis")
        rl_vs_zy.append(rl_zy_dist[0,0])
        subject_list.append(s)
        continue
rl_zy_fore_df=pd.DataFrame({"subject":subject_list, "rl_vs_zy":rl_vs_zy})


#combine different comparisons with sequential merge
merge1_qpcr=ps_rl_fore_df.merge(ps_zy_fore_df, how="outer", on="subject")
bcdist_fore_withinsub_methods_qpcr=merge1_qpcr.merge(rl_zy_fore_df, how="outer", on="subject")

#legs
data=qpcr_skin_meta[qpcr_skin_meta["source"]=="leg"].fillna(0)
#ps vs rl
ps_vs_rl=[]
subject_list=[]
for s in data["subject"].unique():
    psdata=np.array(data[(data["subject"]==s)&(data["method"]=="powersoil")].iloc[:,:-16].apply(pd.to_numeric))
    rldata=np.array(data[(data["subject"]==s)&(data["method"]=="readylyse_dil")].iloc[:,:-16].apply(pd.to_numeric))
    if len(psdata)!=0 and len(rldata)!=0:
        ps_rl_dist=scipy.spatial.distance.cdist(psdata, rldata,metric="braycurtis")
        ps_vs_rl.append(ps_rl_dist[0,0])
        subject_list.append(s)
        continue
ps_rl_leg_df=pd.DataFrame({"subject":subject_list, "ps_vs_rl":ps_vs_rl})

#plot bc dist comparing extraction methods in pairwise fashion

#melt forehead df easier viz
melt_fore_qpcr_method_bc=bcdist_fore_withinsub_methods_qpcr.melt(id_vars="subject")

#split by source but show on same y axis
# not included as a figure but included as code for interest #

fig,axs=plt.subplots(1,2,sharey=True, figsize=(6,3))
#plot 1 forehead
df_med1=melt_fore_qpcr_method_bc.groupby("variable",sort=True)["value"].median()
df_med1=df_med1.reindex(["ps_vs_zy", "ps_vs_rl", "rl_vs_zy"])
p1=sns.stripplot(ax=axs[0], data=melt_fore_qpcr_method_bc, x="variable", y="value", color="#aba2c6", order=["ps_vs_zy", "ps_vs_rl", "rl_vs_zy"], linewidth=1, edgecolor="gray")
_ = [p1.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_med1.reset_index()["value"].items()]
axs[0].set_xticks([])
axs[0].set_ylabel("")
axs[0].set_xlabel("")
axs[0].spines[['top','right']].set_visible(False)
#plot 2 leg
df_med2=ps_rl_leg_df["ps_vs_rl"].median()
p2=sns.stripplot(ax=axs[1], data=ps_rl_leg_df, y="ps_vs_rl", color="#aba2c6", linewidth=1, edgecolor="gray")
_ = p2.hlines(df_med2, 0-.5, 0+0.5, zorder=2, color="black")
axs[1].set_xticks([])
axs[1].set_xlabel("")
axs[1].spines[['top', 'left','right']].set_visible(False)
plt.savefig(work_dir+"/main_figures/Fig4B_bcdist_methods_qPCRdata.svg", format="svg")
plt.show()

#Mann whitney test for statistics shown on figure
mw_ps_zy_vs_ps_rl_fore_q=stats.mannwhitneyu(bcdist_fore_withinsub_methods_qpcr["ps_vs_zy"].dropna(), bcdist_fore_withinsub_methods_qpcr["ps_vs_rl"].dropna())
mw_ps_zy_vs_rl_zy_fore_q=stats.mannwhitneyu(bcdist_fore_withinsub_methods_qpcr["ps_vs_zy"].dropna(), bcdist_fore_withinsub_methods_qpcr["rl_vs_zy"].dropna())
mw_ps_rl_vs_rl_zy_fore_q=stats.mannwhitneyu(bcdist_fore_withinsub_methods_qpcr["ps_vs_rl"].dropna(), bcdist_fore_withinsub_methods_qpcr["rl_vs_zy"].dropna())

# #visualize samples with very high BC distance across methods
# highleg=ps_rl_leg_df[ps_rl_leg_df["ps_vs_rl"]>0.5]["subject"]
# plotme=qpcr_skin_meta[(qpcr_skin_meta["source"]=="leg")&(qpcr_skin_meta["subject"]==14)]
# colsum=plotme.iloc[:,:-16].sum(axis=0)
# includecol=colsum[colsum>0].index
# plotme[includecol].plot.bar(stacked=True)
# plt.legend(bbox_to_anchor=(1.0,1.0))
# plt.show()
#%% Figure 4: comparing different extraction methods holding DNA analysis method constant 

## main text: qPCR  ##
 #comparing across methods - need to subset just the skin panel used for all three extraction methods
     #dataset filtered for good samples and taxa threshold and removed F magna and renormalized
     # use me: nofm_full_qpcr_skin_renorm
     # 4A is above - universal 16S qPCR

## Fig 4B: Simpson diversity by method using qPCR data
#only showing forehead - has different yield by method
#this df includes all in case you want to visualize other sites

qpcr_simpson=alpha_diversity(metric="simpson", counts=nofm_full_qpcr_skin_renorm.iloc[:,:-1], ids=nofm_full_qpcr_skin_renorm["sample-id"])
qpcr_counts=alpha_diversity(metric="sobs", counts=nofm_full_qpcr_skin_renorm.iloc[:,:-1], ids=nofm_full_qpcr_skin_renorm["sample-id"])
qpcr_alpha=pd.concat([qpcr_simpson,qpcr_counts],axis=1,keys=["qpcr_simpson", "qpcr_counts"])
qpcr_alpha["sample-id"]=qpcr_alpha.index
qpcr_alpha=qpcr_alpha.reset_index(drop=True)
#add metadata
alpha_div_qpcr_meta=qpcr_alpha.merge(meta, how="inner", on="sample-id")

#forehead alpha, plotting simpson 
alpha_div_qpcr_fore=alpha_div_qpcr_meta[alpha_div_qpcr_meta["source"]=="forehead"]
df_median=alpha_div_qpcr_fore.groupby('method', sort=True)["qpcr_simpson"].median()
df_median=df_median.reindex(["powersoil", "readylyse_dil", "zymobiomics"])
plt.rcParams["figure.figsize"] = (4,4)
p=sns.stripplot(data=alpha_div_qpcr_fore, x="method", y="qpcr_simpson",order=["powersoil", "readylyse_dil", "zymobiomics"],s=8,hue="method", palette=method_pal, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["qpcr_simpson"].items()]
plt.ylim(0,1)
plt.xticks([])
plt.ylabel("")
plt.xlabel("")
plt.savefig(work_dir+"/supp_figures/FigS4A_fore_simpson_qpcr.png", format="png")
plt.show()
#stats for fore simpson
mw_ps_rl=stats.mannwhitneyu(alpha_div_qpcr_fore[alpha_div_qpcr_fore["method"]=="powersoil"]["qpcr_simpson"], alpha_div_qpcr_fore[alpha_div_qpcr_fore["method"]=="readylyse_dil"]["qpcr_simpson"])
mw_ps_zy=stats.mannwhitneyu(alpha_div_qpcr_fore[alpha_div_qpcr_fore["method"]=="powersoil"]["qpcr_simpson"].dropna(), alpha_div_qpcr_fore[alpha_div_qpcr_fore["method"]=="zymobiomics"]["qpcr_simpson"].dropna())
mw_rl_zy=stats.mannwhitneyu(alpha_div_qpcr_fore[alpha_div_qpcr_fore["method"]=="zymobiomics"]["qpcr_simpson"].dropna(), alpha_div_qpcr_fore[alpha_div_qpcr_fore["method"]=="readylyse_dil"]["qpcr_simpson"].dropna())


##Supplemental Figure S8: comparing methods for leg samples using qPCR data
## do not include zymo samples too few for accurate analysis ##
#leg alpha, plotting simpson 
alpha_div_qpcr_leg=alpha_div_qpcr_meta[alpha_div_qpcr_meta["source"]=="leg"]
df_median=alpha_div_qpcr_leg.groupby('method', sort=True)["qpcr_simpson"].median()
df_median=df_median.reindex(["powersoil", "readylyse_dil"])
plt.rcParams["figure.figsize"] = (4,4)
p=sns.stripplot(data=alpha_div_qpcr_leg, x="method", y="qpcr_simpson",order=["powersoil", "readylyse_dil"],s=8,hue="method", palette=method_pal, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["qpcr_simpson"].items()]
plt.ylim(0,1)
plt.xticks([])
plt.ylabel("")
plt.xlabel("")
plt.savefig(work_dir+"/supp_figures/FigS4A_leg_simpson_qpcr.png", format="png")
plt.show()
#stats for leg simpson
mw_ps_rl_leg=stats.mannwhitneyu(alpha_div_qpcr_leg[alpha_div_qpcr_leg["method"]=="powersoil"]["qpcr_simpson"].dropna(), alpha_div_qpcr_leg[alpha_div_qpcr_leg["method"]=="readylyse_dil"]["qpcr_simpson"].dropna())

# leg beta 
#just rl and ps samples 
legsamplelist=meta[(meta["source"]=="leg")&(meta["method"]!="zymobiomics")]["sample-id"]
leg_qpcr_skin=nofm_full_qpcr_skin_renorm[nofm_full_qpcr_skin_renorm["sample-id"].isin(legsamplelist)]
#calculate BC distance
leg_bc=squareform(pdist(leg_qpcr_skin.iloc[:,:-1].fillna(0), metric='braycurtis'))

#perform PCA
pca = PCA(n_components=10)
pcoa_leg = pd.DataFrame(pca.fit_transform(leg_bc))
pcoa_leg=pd.concat([pcoa_leg,leg_qpcr_skin["sample-id"].reset_index(drop=True)],axis=1)
pcoa_leg_meta=pcoa_leg.merge(meta, how="inner", on="sample-id")
ratio=pd.Series(pca.explained_variance_ratio_)
plt.rcParams["figure.figsize"] = (4,4)
sns.scatterplot(data=pcoa_leg_meta, x=0, y=1, hue="method", style="method", palette=method_pal, edgecolor="black", s=50, alpha=0.8)
plt.xlabel('PCo1 '+str(np.round(ratio[0],2)*100)+"% explained")
plt.ylabel('PCo2 '+str(np.round(ratio[1],2)*100)+'% explained')
plt.xlabel("")
plt.ylabel("")
#manually generate legend with correct colors
ps=mlines.Line2D([0], [0], marker='o', color='w', label="PowerSoil", markerfacecolor="#465e70", markersize=8)
rl=mlines.Line2D([0], [0], marker='X', color='w', label="ReadyLyse", markerfacecolor="#6587a1", markersize=8)
plt.legend(handles=[ps,rl])
plt.savefig(work_dir+"/supp_figures/FigS4C_leg_bc_dist_qpcr_pcoa.png", format="png")
#plt.title('Forehead PCoA with Bray-Curtis')
plt.show()

## chekcing for stats ##
#using PERMANOVA to add stats and see if controls are separate from samples

leg_method_qpcr_dm=distance.DistanceMatrix(leg_bc, ids=leg_qpcr_skin["sample-id"])
pcoa_leg_meta.index=pcoa_leg_meta["sample-id"]
print(distance.permanova(leg_method_qpcr_dm, pcoa_leg_meta["method"]))


## Fig 4C: Beta diversity by method using qPCR data ##

#only forehead included in paper as figure
#Note: using PCA on bray curtis DM = PCoA ##
#same dataset as used for alpha div, split by source
foresamplelist=meta[meta["source"]=="forehead"]["sample-id"]
fore_qpcr_skin=nofm_full_qpcr_skin_renorm[nofm_full_qpcr_skin_renorm["sample-id"].isin(foresamplelist)]

#calculate BC distance
forehead_bc=squareform(pdist(fore_qpcr_skin.iloc[:,:-1].fillna(0), metric='braycurtis'))

#perform PCA
pca = PCA(n_components=10)
pcoa_forehead = pd.DataFrame(pca.fit_transform(forehead_bc))
pcoa_forehead=pd.concat([pcoa_forehead,fore_qpcr_skin["sample-id"].reset_index(drop=True)],axis=1)
pcoa_fore_meta=pcoa_forehead.merge(meta, how="inner", on="sample-id")
ratio=pd.Series(pca.explained_variance_ratio_)
plt.rcParams["figure.figsize"] = (4,4)
sns.scatterplot(data=pcoa_fore_meta, x=0, y=1, hue="method", style="method", palette=method_pal, edgecolor="black", s=50, alpha=0.8)
plt.xlabel('PCo1 '+str(np.round(ratio[0],2)*100)+"% explained")
plt.ylabel('PCo2 '+str(np.round(ratio[1],2)*100)+'% explained')
plt.xlabel("")
plt.ylabel("")
#manually generate legend with correct colors
ps=mlines.Line2D([0], [0], marker='o', color='w', label="PowerSoil", markerfacecolor="#465e70", markersize=8)
rl=mlines.Line2D([0], [0], marker='X', color='w', label="ReadyLyse", markerfacecolor="#6587a1", markersize=8)
zy=mlines.Line2D([0], [0], marker='s', color='w', label="ZymoBIOMICS", markerfacecolor="#b2c3d0", markersize=8)
plt.legend(handles=[ps,rl, zy])
plt.savefig(work_dir+"/supp_figures/FigS4C_fore_bc_dist_qpcr_pcoa.png", format="png")
#plt.title('Forehead PCoA with Bray-Curtis')
plt.show()

## chekcing for stats ##
#using PERMANOVA to add stats and see if controls are separate from samples

fore_method_qpcr_dm=distance.DistanceMatrix(forehead_bc, ids=fore_qpcr_skin["sample-id"])
pcoa_fore_meta.index=pcoa_fore_meta["sample-id"]
print(distance.permanova(fore_method_qpcr_dm, pcoa_fore_meta["method"]))


# ## Fig. 4D: mock community composition by different DNA extraction methods qPCR
# species level filtered data (good samples, no F. magna): nofm_full_qpcr_skin_renorm

#add metadata
qpcr_skin_filt_meta=nofm_full_qpcr_skin_renorm.merge(meta, how="inner", on="sample-id")

#splitting by method
mock_ps=qpcr_skin_filt_meta[(qpcr_skin_filt_meta["sampletype"]=="lm_mock")&(qpcr_skin_filt_meta["method"]=="powersoil")].iloc[:,:-16]
mock_rl=qpcr_skin_filt_meta[(qpcr_skin_filt_meta["sampletype"]=="lm_mock")&(qpcr_skin_filt_meta["method"]=="readylyse_dil")].iloc[:,:-16]
mock_zy=qpcr_skin_filt_meta[(qpcr_skin_filt_meta["sampletype"]=="lm_mock")&(qpcr_skin_filt_meta["method"]=="zymobiomics")].iloc[:,:-16]

#combine into a single df
#include mock hypothetical community 
allmethod_mock=pd.concat([mock_hypo, mock_ps,mock_rl, mock_zy,]).fillna(0)
#drop extra hypothetical rows
allmethod_mock2=allmethod_mock.iloc[2:,1:]
#only include columns with values above 1 perc or 0.01
coltotal=allmethod_mock2.sum(axis=0)
includecol=coltotal[coltotal>0.01].index
allmethod_mock2=allmethod_mock2[includecol]
allmethod_mock2["other"]=1-allmethod_mock2.sum(axis=1)
#add spacer to distinguish between methods
#also keeping a blank space for the readylyse mock which had no signal
allmethod_mock2["spacer"]=[1,3,4,5,7,8,10,12,13,14,15]
spacelist=np.arange(1,16)
allmethod_mock2=allmethod_mock2.merge(pd.DataFrame({"spacer":spacelist}), on="spacer", how="right")
#add metadata for sanity
#allmethod_mock2["method"]=["hypo","ps", "ps", "ps", "rl", "rl", "rl", "rl", "zy", "zy", "zy", "zy"]
allmethod_mock2.loc['ave']=allmethod_mock2.iloc[:,:-2].mean(axis=0)
#plot figure
plt.rcParams["figure.figsize"] = (5,3.5)
allmethod_mock2=allmethod_mock2.sort_values(by="ave", axis=1, ascending=False).iloc[:-1,:]
ax=allmethod_mock2.iloc[:,:-2].plot.bar(stacked=True, color=taxa_pal_sg_pmp, width=0.9, linewidth=1, edgecolor="gray")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.ylim(0,1)
plt.ylabel("")
plt.xticks([])
plt.savefig(work_dir+"/supp_figures/FigS4E_mock_barplot_by_extraction_method_qpcr.svg", format="svg")
plt.show()


##### supplemental figure including untargeted but lower resoluation 16S data ####
## Fig. S4: Simpson diversity by method using 16S ##

 #comparing across methods 
     #dataset filtered for good samples and taxa threshold and removed F magna and renormalized
     # use me: amplicon_nof_renorm

amp_all_methods_simpson=alpha_diversity(metric="simpson", counts=amplicon_nof_renorm.iloc[:,:-2], ids=amplicon_nof_renorm["sample-id"])
amp_all_methods_counts=alpha_diversity(metric="sobs", counts=amplicon_nof_renorm.iloc[:,:-2], ids=amplicon_nof_renorm["sample-id"])
amp_allmethods_alpha=pd.concat([amp_all_methods_simpson,amp_all_methods_counts],axis=1,keys=["amp_simpson", "amp_counts"])
amp_allmethods_alpha["sample-id"]=amp_allmethods_alpha.index
amp_allmethods_alpha=amp_allmethods_alpha.reset_index(drop=True)
#add metadata
amp_allmethods_alpha_meta=amp_allmethods_alpha.merge(meta, how="inner", on="sample-id")

#forehead alpha, plotting simpson 
alpha_div_amp_fore=amp_allmethods_alpha_meta[amp_allmethods_alpha_meta["source"]=="forehead"]
df_median=alpha_div_amp_fore.groupby('method', sort=True)["amp_simpson"].median()
df_median=df_median.reindex(["powersoil", "readylyse_dil", "zymobiomics"])
plt.rcParams["figure.figsize"] = (4,4)
p=sns.stripplot(data=alpha_div_amp_fore, x="method", y="amp_simpson",order=["powersoil", "readylyse_dil", "zymobiomics"],s=8,hue="method", palette=method_pal, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["amp_simpson"].items()]
plt.ylim(0,1)
plt.xticks([])
plt.ylabel("")
plt.xlabel("")
plt.savefig(work_dir+"/supp_figures/FigS4B_fore_simpson_amplicon.png", format="png")
plt.show()
#stats for fore simpson
mw_ps_rl_a=stats.mannwhitneyu(alpha_div_amp_fore[alpha_div_amp_fore["method"]=="powersoil"]["amp_simpson"], alpha_div_amp_fore[alpha_div_amp_fore["method"]=="readylyse_dil"]["amp_simpson"])
mw_ps_zy_a=stats.mannwhitneyu(alpha_div_amp_fore[alpha_div_amp_fore["method"]=="powersoil"]["amp_simpson"].dropna(), alpha_div_amp_fore[alpha_div_amp_fore["method"]=="zymobiomics"]["amp_simpson"].dropna())
mw_rl_zy_a=stats.mannwhitneyu(alpha_div_amp_fore[alpha_div_amp_fore["method"]=="zymobiomics"]["amp_simpson"].dropna(), alpha_div_amp_fore[alpha_div_amp_fore["method"]=="readylyse_dil"]["amp_simpson"].dropna())

#leg alpha plotting simpson 
alpha_div_amp_leg=amp_allmethods_alpha_meta[amp_allmethods_alpha_meta["source"]=="leg"]
df_median=alpha_div_amp_leg.groupby('method', sort=True)["amp_simpson"].median()
df_median=df_median.reindex(["powersoil", "readylyse_dil", "zymobiomics"])
plt.rcParams["figure.figsize"] = (4,4)
p=sns.stripplot(data=alpha_div_amp_leg, x="method", y="amp_simpson",order=["powersoil", "readylyse_dil", "zymobiomics"],s=8,hue="method", palette=method_pal, linewidth=1, edgecolor="gray")
_ = [p.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["amp_simpson"].items()]
plt.ylim(0,1)
plt.xticks([])
plt.ylabel("")
plt.xlabel("")
plt.savefig(work_dir+"/supp_figures/FigS4B_leg_simpson_amplicon.png", format="png")
plt.show()
#stats for fore simpson
mw_ps_rl_a_leg=stats.mannwhitneyu(alpha_div_amp_leg[alpha_div_amp_leg["method"]=="powersoil"]["amp_simpson"], alpha_div_amp_leg[alpha_div_amp_leg["method"]=="readylyse_dil"]["amp_simpson"])
mw_ps_zy_a_leg=stats.mannwhitneyu(alpha_div_amp_leg[alpha_div_amp_leg["method"]=="powersoil"]["amp_simpson"].dropna(), alpha_div_amp_leg[alpha_div_amp_leg["method"]=="zymobiomics"]["amp_simpson"].dropna())
mw_rl_zy_a_leg=stats.mannwhitneyu(alpha_div_amp_leg[alpha_div_amp_leg["method"]=="zymobiomics"]["amp_simpson"].dropna(), alpha_div_amp_leg[alpha_div_amp_leg["method"]=="readylyse_dil"]["amp_simpson"].dropna())

## beta diversity by 16S colored by method ##

fore_amp_skin=amplicon_nof_renorm[amplicon_nof_renorm["sample-id"].isin(foresamplelist)]

#calculate BC distance
forehead_bc_a=squareform(pdist(fore_amp_skin.iloc[:,:-2].fillna(0), metric='braycurtis'))

#perform PCA
pca = PCA(n_components=10)
pcoa_forehead_a = pd.DataFrame(pca.fit_transform(forehead_bc_a))
pcoa_forehead_a=pd.concat([pcoa_forehead_a,fore_amp_skin["sample-id"].reset_index(drop=True)],axis=1)
pcoa_fore_meta_a=pcoa_forehead_a.merge(meta, how="inner", on="sample-id")
ratio=pd.Series(pca.explained_variance_ratio_)
plt.rcParams["figure.figsize"] = (4,4)
sns.scatterplot(data=pcoa_fore_meta_a, x=0, y=1, hue="method", style="method", palette=method_pal, edgecolor="gray")
plt.xlabel('PCo1 '+str(np.round(ratio[0],2)*100)+"% explained")
plt.ylabel('PCo2 '+str(np.round(ratio[1],2)*100)+'% explained')
plt.xlabel("")
plt.ylabel("")
#manually generate legend with correct colors
ps=mlines.Line2D([0], [0], marker='o', color='w', label="PowerSoil", markerfacecolor="#465e70", markersize=8)
rl=mlines.Line2D([0], [0], marker='X', color='w', label="ReadyLyse", markerfacecolor="#6587a1", markersize=8)
zy=mlines.Line2D([0], [0], marker='s', color='w', label="ZymoBIOMICS", markerfacecolor="#b2c3d0", markersize=8)
plt.legend(handles=[ps,rl, zy])
plt.savefig(work_dir+"/supp_figures/FigS4D_fore_bc_dist_pcoa_amp.png", format="png")
#plt.title('Forehead PCoA with Bray-Curtis')
plt.show()

fore_method_16s_dm=distance.DistanceMatrix(forehead_bc_a, ids=fore_amp_skin["sample-id"])
pcoa_fore_meta_a.index=pcoa_fore_meta_a["sample-id"]
print(distance.permanova(fore_method_16s_dm, pcoa_fore_meta_a["method"]))

#leg 16S beta diversity 
alllegsamples=meta[meta["source"]=="leg"]["sample-id"]
leg_amp_skin=amplicon_nof_renorm[amplicon_nof_renorm["sample-id"].isin(alllegsamples)]

#calculate BC distance
leg_bc_a=squareform(pdist(leg_amp_skin.iloc[:,:-2].fillna(0), metric='braycurtis'))

#perform PCA
pcoa_leg_a = pd.DataFrame(pca.fit_transform(leg_bc_a))
pcoa_leg_a=pd.concat([pcoa_leg_a,leg_amp_skin["sample-id"].reset_index(drop=True)],axis=1)
pcoa_leg_meta_a=pcoa_leg_a.merge(meta, how="inner", on="sample-id")
ratio=pd.Series(pca.explained_variance_ratio_)
plt.rcParams["figure.figsize"] = (4,4)
sns.scatterplot(data=pcoa_leg_meta_a, x=0, y=1, hue="method", style="method", palette=method_pal, edgecolor="gray")
plt.xlabel('PCo1 '+str(np.round(ratio[0],2)*100)+"% explained")
plt.ylabel('PCo2 '+str(np.round(ratio[1],2)*100)+'% explained')
plt.xlabel("")
plt.ylabel("")
#manually generate legend with correct colors
ps=mlines.Line2D([0], [0], marker='o', color='w', label="PowerSoil", markerfacecolor="#465e70", markersize=8)
rl=mlines.Line2D([0], [0], marker='X', color='w', label="ReadyLyse", markerfacecolor="#6587a1", markersize=8)
zy=mlines.Line2D([0], [0], marker='s', color='w', label="ZymoBIOMICS", markerfacecolor="#b2c3d0", markersize=8)
plt.legend(handles=[ps,rl, zy])
plt.savefig(work_dir+"/supp_figures/FigS4B_leg_bc_dist_pcoa_amp.png", format="png")
#plt.title('Forehead PCoA with Bray-Curtis')
plt.show()

leg_method_16s_dm=distance.DistanceMatrix(leg_bc_a, ids=leg_amp_skin["sample-id"])
pcoa_leg_meta_a.index=pcoa_leg_meta_a["sample-id"]
print(distance.permanova(leg_method_16s_dm, pcoa_leg_meta_a["method"]))

## mock community composition by 16S split by method ##
#add metadata
amplicon_nof_meta=amplicon_nof_renorm.merge(meta, how="inner", on="sample-id")

#splitting by method
mock_ps_a=amplicon_nof_meta[(amplicon_nof_meta["sampletype"]=="lm_mock")&(amplicon_nof_meta["method"]=="powersoil")].iloc[:,:-17]
mock_rl_a=amplicon_nof_meta[(amplicon_nof_meta["sampletype"]=="lm_mock")&(amplicon_nof_meta["method"]=="readylyse_dil")].iloc[:,:-17]
mock_zy_a=amplicon_nof_meta[(amplicon_nof_meta["sampletype"]=="lm_mock")&(amplicon_nof_meta["method"]=="zymobiomics")].iloc[:,:-17]

#combine into a single df
#include mock hypothetical community at the genus level
#add hypothetical input
mock_hypo_genus=mock_hypo.iloc[:,:6]
genus_columns=["sample", "Corynebacterium", "Staphylococcus", "Staphylococcus", "Escherichia", "Cutibacterium"]
mock_hypo_genus.columns=genus_columns
#sum staphylococcus
mock_hypo_genus["sum_staph"]=mock_hypo_genus["Staphylococcus"].sum(axis=1)
mockhypo2=mock_hypo_genus.drop(columns=["Staphylococcus"])
mockhypo2=mockhypo2.rename(columns={"sum_staph":"Staphylococcus"})
#drop extra hypothetical rows
mockhypo3=mockhypo2.iloc[:-2,1:]

allmethod_mock_a=pd.concat([mockhypo3, mock_ps_a,mock_rl_a, mock_zy_a,], axis=0).fillna(0)

#only include columns with values above 1 perc or 0.01
coltotal=allmethod_mock_a.sum(axis=0)
includecol=coltotal[coltotal>0.01].index
allmethod_mock2_a=allmethod_mock_a[includecol]
allmethod_mock2_a["other"]=1-allmethod_mock2_a.sum(axis=1)
#add metadata for sanity
allmethod_mock2_a["method"]=["hypo","ps", "ps", "ps", "rl", "rl", "rl", "rl", "zy", "zy", "zy", "zy"]
#add spacer to distinguish between methods
allmethod_mock2_a["spacer"]=[1,3,4,5,7,8,9,10,12,13,14,15]
spacelist=np.arange(1,16)
allmethod_mock2_a=allmethod_mock2_a.merge(pd.DataFrame({"spacer":spacelist}), on="spacer", how="right")
allmethod_mock2_a.loc['ave']=allmethod_mock2_a.iloc[:,:-2].mean(axis=0)
#plot figure
plt.rcParams["figure.figsize"] = (5,3.5)
allmethod_mock2_a=allmethod_mock2_a.sort_values(by="ave", axis=1, ascending=False).iloc[:-1,:]
ax=allmethod_mock2_a.iloc[:,:-2].plot.bar(stacked=True, color=genus_pal, width=0.9, linewidth=1, edgecolor="gray")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.ylim(0,1)
plt.ylabel("")
plt.xticks([])
plt.savefig(work_dir+"/supp_figures/SFig4E_mock_barplot_16s_by_method.svg", format="svg")
plt.show()
#%% Figure 5: interesting species-level differences between subjects shotgun metagenomics

  ### Fig 5A: species level barplot shotgun metagenomics
  #all leg samples sorted by sex then by gardnerella abundance
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma']
leg_sg=sg_col_meta[sg_col_meta["source"]=="leg"]
#leg_sg["subject"]=leg_sg["subject"].astype("str")
#manually sort samples so that they can be both sorted by sex and Gardnerella vaginalis with all the female samples in sequence
manorder_legs=[4,8,9,14,16,7,6,19,12,3,5,11,17]
leg_sg.loc["ave"]=leg_sg.iloc[:,:-16].mean()
leg_sg=leg_sg.sort_values(by="ave", axis=1, ascending=False).iloc[:-1,:]
#subset out gray columns to put at end of list
graycol=hexcolors_sg_pmp[hexcolors_sg_pmp["color"]=="gray"]["taxa"]
graycol_dataset=pd.Series(leg_sg.columns[leg_sg.columns.isin(graycol)])
#subset out color columns
colorcol=pd.Series(leg_sg.columns[:-16][~leg_sg.columns[:-16].isin(graycol)])
#save file and manually reorder to sort by genus
colorcol.to_csv(work_dir+"/data/hex_color_keys/sg_meta_color_to_sort.csv")
#read in sorted file
colornames=pd.read_csv(work_dir+"/data/hex_color_keys/reorder_sg_leg_rev.csv", header=None)
neworder=pd.concat([colornames[0],graycol_dataset])
neworder=neworder[neworder!="sample-id"]
#sort by given subject order manorder
leg_sg=leg_sg.set_index('subject').reindex(manorder_legs).reset_index()
plt.rcParams["figure.figsize"] = (6,4)
fig, axs =plt.subplots()
leg_sg[neworder].plot.bar(ax=axs, stacked=True, color=taxa_pal_sg_pmp,width=0.8, linewidth=1, edgecolor="black")
L=axs.legend()
plt.setp(L.texts, family='Consolas')
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xticks([])
axs.spines[['top','right']].set_visible(False)
plt.ylim(0,1)
#plt.xticks(np.arange(0,len(manorder)), leg_sg["subject"])
#plt.title("Shotgun metagenomics: legs")
plt.savefig(work_dir+"/main_figures/Fig5A_leg_metgenomics_barplot.svg", format="svg")
plt.show()

    ###Fig S9: forehead composition in same order as legs
manorder_fore=[4, 8, 9, 14, 16, 7, 6, 19, 15, 12, 20, 18, 3, 5, 11, 17]
fore_sg=sg_col_meta[sg_col_meta["source"]=="forehead"]
fore_sg.loc["ave"]=fore_sg.iloc[:,:-16].mean()
fore_sg=fore_sg.sort_values(by="ave", axis=1, ascending=False).iloc[:-1,:]
#subset out gray columns to put at end of list
graycol=hexcolors_sg_pmp[hexcolors_sg_pmp["color"]=="gray"]["taxa"]
graycol_dataset=pd.Series(fore_sg.columns[fore_sg.columns.isin(graycol)])
#subset out color columns
#colorcol=pd.Series(fore_sg.columns[:-16][~fore_sg.columns[:-16].isin(graycol)])
#read in file to get order of color columns 
color_fore_names=pd.read_csv(work_dir+"/data/hex_color_keys/forehead_sp_order.csv", header=None)
neworder=pd.concat([color_fore_names[0],graycol_dataset])
neworder=neworder[neworder!="sample-id"]
#sort by given subject order manorder
fore_sg=fore_sg.set_index('subject').reindex(manorder_fore).reset_index()
fig, axs =plt.subplots()
fore_sg[neworder].plot.bar(ax=axs, stacked=True, color=taxa_pal_sg_pmp,width=0.8, linewidth=1, edgecolor="black")
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xticks([])
axs.spines[['top','right']].set_visible(False)
plt.ylim(0,1)
#plt.xticks(np.arange(0,len(manorder)), fore_sg["subject"])
#plt.title("Shotgun metagenomics: forehead")
plt.savefig(work_dir+"/supp_figures/FigS10A_forehead_metgenomics_barplot.svg", format="svg", bbox_inches="tight")
plt.show()

  ### Fig 5B: qPCR abundance of Gardnerella vaginalis ###
  # using sample-filtered, no F magna data set although doesn't really matter becuase abs abundance: qpcr_abs_abun_nofm2
color14=pd.read_csv(work_dir+"/data/hex_color_keys/color14pal.csv")
subjectpal2=dict(zip(color14["subj"], color14["color"]))

#add metadata to qpcr dataset
qpcr_abs_meta=qpcr_abs_abun_nofm2.merge(meta, how="inner", on="sample-id")
powersoil_qpcr=qpcr_abs_meta[qpcr_abs_meta["method"]=="powersoil"]
swabstoplot=powersoil_qpcr[(powersoil_qpcr["source"]=="leg")|(powersoil_qpcr["source"]=="forehead")]
#sample with g vaginalis above 0
plotgard=swabstoplot[swabstoplot["Gardnerella vaginalis"]>0]["sample-id"]
plt.rcParams["figure.figsize"] = (3,3)
p=sns.stripplot(data=swabstoplot[swabstoplot["sample-id"].isin(plotgard)], x="source", y="Gardnerella vaginalis",hue="subject", palette=subjectpal2,  linewidth=0.8, edgecolor="gray", size=8, jitter=True)
plt.yscale('log')
plt.legend(bbox_to_anchor=(1.0,1.0), title="Subject")
plt.ylabel("")
plt.xlabel("")
p.spines[['top', 'right']].set_visible(False)
plt.savefig(work_dir+"/main_figures/Fig5C_qpcr_abs_gvag.png", format="png", bbox_inches="tight")
plt.show()

#%%PHLAME strain analysis of C acnes
#use Evan code to combine his analysis into plot-able dataframes
#read in PHLAME output: sample list and phylogroup frequencies
cacnes_samples=pd.read_csv(work_dir+"/data/PHLAME_analysis/samples.csv")["Sample"]
cacnes_out_freq=hf.read_phlame_frequencies_NEW(cacnes_samples, work_dir+"/data/PHLAME_analysis/6-frequencies", "Pacnes_C1")

#fix sample names
#truly insane string splitting because the samples.csv has the dumb names I used once
sample_id=cacnes_samples.str[-3:]
sample_id=sample_id.replace("097", "97")
#create condition and choice list to identify dilution of beads used based on location 
cond_list=[cacnes_samples.str.contains("forehead"), cacnes_samples.str.contains("leg"), cacnes_samples.str.contains("mock"), cacnes_samples.str.contains("buffer"), cacnes_samples.str.contains("air")]
choice=[cacnes_samples.str[8:10], cacnes_samples.str[3:5], cacnes_samples.str[5:7], cacnes_samples.str[6:8], cacnes_samples.str[3:5]]
choice2=["forehead", "leg", "mock", "buffer", "air"]
sample_dil=np.select(cond_list, choice, "None")
sample_source=np.select(cond_list, choice2, "other")
#recombine with data
cacnes_id=pd.DataFrame({"sample":cacnes_samples, "sample-id":sample_id.astype("int64"),"tag_bead_dil":sample_dil})

#read in list of phylogroup IDs from Evan
phylogroups = np.loadtxt(work_dir+'/data/PHLAME_analysis/Cacnes_megatree_phylogroupIDs.txt',dtype=str)
#subset out only phylogroup level frequencies
phylogroup_frequencies = cacnes_out_freq[np.unique(phylogroups[:,2])]
phylogroup_frequencies.columns = [np.unique(phylogroups[np.where(phylogroups[:,2] == clade),1])[0] for clade in np.unique(phylogroups[:,2])]
uncorr_sum=phylogroup_frequencies.sum(axis=1)
#add "no call" column and renormalize frequencies
norm_to_1 = phylogroup_frequencies.loc[(phylogroup_frequencies.sum(axis=1) > 1)]
phylogroup_frequencies.loc[(phylogroup_frequencies.sum(axis=1) > 1)] = norm_to_1.div(norm_to_1.sum(axis=1), axis=0)

#add metadata for plottingg ease
phylogroup_frequencies["sample"]=phylogroup_frequencies.index 
phylogroup_samples=phylogroup_frequencies.merge(cacnes_id, how="inner", on="sample")
phylogroup_meta=phylogroup_samples.merge(meta, how='inner', on="sample-id")
#only using the 1:50 dilution samples
phylogroup_meta=phylogroup_meta[phylogroup_meta["tag_bead_dil"]=="50"]

#remove samples which were splash contaminated or too low reads
phylogroup_meta=phylogroup_meta[~phylogroup_meta["sample-id"].isin(dropsample)]
phylogroup_meta=phylogroup_meta[~phylogroup_meta["sample-id"].isin(drop_sg_samples["sample-id"])]

phylogroup_meta.to_csv(work_dir+"/data/PHLAME_analysis/cacnes_phylogroup_nodups.csv", index=None)

#visualize abundance of PHylogroups: Fig. S10

    ###Fig S9C ###
    
#show that absolute abund varies between leg and forehead but both have diverse phylotypes like real samples
phylo_pal=dict(Undefined="white", A="#d2d2cb", B="#4d695d", C="#83a79d", D="#7393B3", E="#30b0e0", F="#a1cde5", H="#abdda4", L="gray", K="#66c2a5")
# leg_phyloave=phylogroup_meta[phylogroup_meta["source"]=="leg"].iloc[:,:-19].mean()
# fore_phyloave=phylogroup_meta[phylogroup_meta["source"]=="forehead"].iloc[:,:-19].mean()
# blankave=phylogroup_meta[(phylogroup_meta["source"]=="air")|(phylogroup_meta["source"]=="buffer")].iloc[:,:-19].mean()
# phylo_ave_df=pd.concat([leg_phyloave, fore_phyloave, blankave],axis=1)
# phylo_ave_df.columns=["leg", "forehead", "blanks"]
# plotavephylo=phylo_ave_df.T
# plt.rcParams["figure.figsize"] = (3,3)
# #fig, ax=plt.subplot()
# plotavephylo.loc["ave"]=plotavephylo.mean()
# plotavephylo=plotavephylo.sort_values(by="ave", axis=1, ascending=False).iloc[:-1,:]
# plotavephylo["Undefined"]=1-plotavephylo.sum(axis=1)
# plotavephylo.plot.bar(stacked=True,width=0.9, linewidth=1, edgecolor="gray", color=phylo_pal)
# plt.xticks([0,1,2], ["leg", "forehead", "negative\ncontrols"],rotation=0)
# plt.ylim(0,1)
# #plt.title("Average abundance C. acnes phylogroups")
# #plt.ylabel("Rel abundance")
# plt.legend(bbox_to_anchor=(1.0,1.0), title="C.acnes Phylogroup")
# #ax.spines['top'].set_visible(False)
# #ax.spines['right'].set_visible(False)
# plt.savefig(work_dir+"/supp_figures/FigS10D_ave_phylo_leg_forehead_barplot.svg", format="svg")
# plt.show()


## showing by subject split by site per Tami request ##
fig, axs=plt.subplots(nrows=2,ncols=1,figsize=(6,10))

#legs
phylo_legs=phylogroup_meta[phylogroup_meta["source"]=="leg"].sort_values(by="subject")
#only include samples with more than 25% defined C acnes on legs
includesubject=phylo_legs[phylo_legs.iloc[:,:-19].sum(axis=1)>0.25]["subject"]
phylo_leg_plot=phylo_legs[phylo_legs["subject"].isin(includesubject)].iloc[:,:-19]
phylo_leg_plot["Undefined"]=1-phylo_leg_plot.sum(axis=1)
phylo_leg_plot.plot.bar(ax=axs[0], stacked=True, width=0.9, linewidth=1, edgecolor="gray", color=phylo_pal)
axs[0].set_ylim(0,1)
axs[0].legend(bbox_to_anchor=(1.0,1.0))
axs[0].set_xticks(np.arange(0,len(includesubject)), includesubject)
axs[0].set_title("Leg samples")

#forehead
phylo_fore=phylogroup_meta[phylogroup_meta["source"]=="forehead"].sort_values(by="subject")
phylo_fore_plot=phylo_fore[phylo_fore["subject"].isin(includesubject)].iloc[:,:-19]
phylo_fore_plot["Undefined"]=1-phylo_fore_plot.sum(axis=1)
phylo_fore_plot.plot.bar(ax=axs[1],stacked=True, width=0.9, linewidth=1, edgecolor="gray", color=phylo_pal)
axs[1].set_ylim(0,1)
axs[1].legend(bbox_to_anchor=(1.0,1.0))
axs[1].set_xticks(np.arange(0,len(includesubject)),includesubject)
axs[1].set_title("Forehead samples")
plt.savefig(work_dir+"/supp_figures/FigS10D_alternate_by_subject_cacnes_phylo.svg", format="svg")
plt.show()

   ### Fig. S10B ###
#show average abundance of each phylogroup in leg and forehead samples
#melt data longform for plotting
plt.rcParams["figure.figsize"] = (4,3)
cacnes_phylo_melt=phylogroup_meta.melt(id_vars=["subject", "sample-id", "source", "sampletype"], value_vars=["A", "L", "K", "H", "C", "B", "F", "E", "D"])
sns.barplot(data=cacnes_phylo_melt[cacnes_phylo_melt["sampletype"]=="swab"], x="variable", y="value", hue="source", dodge=True, palette=source_pal, errorbar="se")
#plt.ylabel("Rel abundance of C. acnes")
#plt.xlabel("C acnes phylogroup")
plt.ylabel("")
plt.xlabel("")
plt.ylim(0,0.5)
plt.savefig(work_dir+"/supp_figures/FigS10C_compare_phylo_group_leg_fore.png", format="png")
plt.show()

    # MAIN TEXT Fig. 5C #
#is the BC distance between leg/fore of the same person lower than all combinations?
within_person_bc=[]
subject_bc=[]
for s in phylogroup_meta["subject"].unique():
    leg=np.array(phylogroup_meta[(phylogroup_meta["subject"]==s)&(phylogroup_meta["source"]=="leg")].iloc[:,:-19].apply(pd.to_numeric))
    fore=np.array(phylogroup_meta[(phylogroup_meta["subject"]==s)&(phylogroup_meta["source"]=="forehead")].iloc[:,:-19].apply(pd.to_numeric))
    if len(leg)!=0 and len(fore)!=0:
        within_dist=scipy.spatial.distance.cdist(leg, fore,metric="braycurtis")
        within_person_bc.append(within_dist[0,0])
        subject_bc.append(s)
        continue
#braycurtis distance between all samples between kit pairs
#using fill diagonal with zeros to remove self-self comparisons
#powersoil versus readylyse
bc_dist_legs_fore=scipy.spatial.distance.cdist(np.array(phylogroup_meta[phylogroup_meta["source"]=="leg"].iloc[:,:-19].apply(pd.to_numeric)),np.array(phylogroup_meta[phylogroup_meta["source"]=="forehead"].iloc[:,:-19].apply(pd.to_numeric)),metric="braycurtis")
bc_dist_legs_fore=pd.DataFrame(bc_dist_legs_fore)
np.fill_diagonal(bc_dist_legs_fore.values, 0)
listbc_leg_fore=bc_dist_legs_fore.melt()
listbc_leg_fore=listbc_leg_fore.rename(columns={"value":"all_fore_leg"})
listbc_leg_fore_noself=listbc_leg_fore[listbc_leg_fore>0]
bcdist_alllegs=scipy.spatial.distance.pdist(np.array(phylogroup_meta[phylogroup_meta["source"]=="leg"].iloc[:,:-19].apply(pd.to_numeric)),metric="braycurtis")
bcdist_allforeheads=scipy.spatial.distance.pdist(np.array(phylogroup_meta[phylogroup_meta["source"]=="forehead"].iloc[:,:-19].apply(pd.to_numeric)),metric="braycurtis")
#summary leg distance dataframe
bc_summary=pd.concat([pd.Series(within_person_bc), listbc_leg_fore_noself["all_fore_leg"], pd.Series(bcdist_alllegs, name="bcdist_alllegs"),pd.Series(bcdist_allforeheads, name="bcdist_allforeheads")], axis=1)
bc_summary=bc_summary.rename(columns={0:"within_subject"})
#melt for plotting with median
bc_summary_melt=bc_summary.melt()
df_median=bc_summary_melt.groupby('variable', sort=True)["value"].median()
df_median=df_median.reindex(["within_subject",  "bcdist_alllegs", "bcdist_allforeheads", "all_fore_leg"])
ax=sns.stripplot(data=bc_summary_melt, x="variable", y="value",zorder=0, color="#aba2c6",order=["within_subject",  "bcdist_alllegs", "bcdist_allforeheads", "all_fore_leg"], s=6, linewidth=1, edgecolor="gray", jitter=True)
_ = [ax.hlines(y, i-.25, i+.25, zorder=2, color="black") for i, y in df_median.reset_index()["value"].items()]
plt.legend('',frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.ylim(0,1)
#plt.xticks([0,1,2,3], ["Within Subject\n(leg vs forehead)","Between subjects\n(legs vs legs)", "Between subjects\n(foreheads vs foreheads)", "Between subjects\n(all legs vs all foreheads)"],rotation=30)
plt.xticks([])
#plt.ylabel("Bray-Curtis Distance")
plt.ylabel("")
plt.xlabel("")
#plt.title("Phylogroup composition within subjects versus between")
plt.savefig(work_dir+"/main_figures/Fig5B_bcdist_cacnes.png", format="png")
plt.show()

#is this diff stat sig
fore_leg_mw=stats.mannwhitneyu(bc_summary["within_subject"].dropna(), bc_summary["all_fore_leg"].dropna())
within_vs_all_leg=stats.mannwhitneyu(bc_summary["within_subject"].dropna(), bc_summary["bcdist_alllegs"].dropna())
all_leg_vs_all_fore=stats.mannwhitneyu(bc_summary["bcdist_alllegs"].dropna(), bc_summary["bcdist_allforeheads"].dropna())
#%% SUPPLEMENTAL FIGURES BELOW ####

## some supplemental figures are above, if they were the same visualization as a main text fig (just a diff dataset)
## below are figures that are in the supplement but don't have main text counterparts 

#%% Supp Fig 2A: supporting that metagenomics leg samples and blanks are distinct

# #starting dataset: sg_nofm_samplefilt (filtered for taxa and samples metagenomics)
#BC distance of legs and neg controls to look for separattion
# not including foreheads because the large difference b4etween forehead and leg probably drives leg and controls closer
#list of sample-ids that are just leg and buffer and air
leg_neg_id=meta[(meta["source"]=="air")|(meta["source"]=="buffer")|(meta["source"]=="leg")]["sample-id"]
sg_leg_neg=sg_nofm_samplefilt[sg_nofm_samplefilt["sample-id"].isin(leg_neg_id)]
#calculate BC distance
sg_leg_bc_distance=pdist(sg_leg_neg.iloc[:,:-1].fillna(0), metric="braycurtis")
square_bc=squareform(sg_leg_bc_distance)

#perform PCA
pca = PCA(n_components=10)
pcoa_leg_neg = pd.DataFrame(pca.fit_transform(square_bc))
pcoa_leg_neg2=pd.concat([pcoa_leg_neg,sg_leg_neg["sample-id"].reset_index(drop=True)],axis=1)
pcoa_leg_neg_meta=pcoa_leg_neg2.merge(meta, how="inner", on="sample-id")
ratio=pd.Series(pca.explained_variance_ratio_)
#viz PC1 and PC2
sns.scatterplot(data=pcoa_leg_neg_meta, x=0, y=1, hue="source", palette=source_pal, edgecolor="gray")
plt.xlabel('PCo1 '+str(np.round(ratio[0],2)*100)+"% explained")
plt.ylabel('PCo2 '+str(np.round(ratio[1],2)*100)+'% explained')
plt.legend(loc='upper left')
plt.xlabel("")
plt.ylabel("")
plt.savefig(work_dir+"/supp_figures/FigS9A_leg_neg_pcoa.png", format="png")
#plt.title('Forehead PCoA with Bray-Curtis')
plt.show()
#viz PC2 and PC3
sns.scatterplot(data=pcoa_leg_neg_meta, x=1, y=2, hue="source", palette=source_pal, edgecolor="gray")
plt.xlabel('PCo2 '+str(np.round(ratio[1],2)*100)+"% explained")
plt.ylabel('PCo3 '+str(np.round(ratio[2],2)*100)+'% explained')
plt.show()
#viz PC1 and PC3
sns.scatterplot(data=pcoa_leg_neg_meta, x=0, y=2, hue="source", palette=source_pal, edgecolor="gray")
plt.xlabel('PCo1 '+str(np.round(ratio[0],2)*100)+"% explained")
plt.ylabel('PCo3 '+str(np.round(ratio[2],2)*100)+'% explained')
plt.show()

#using PERMANOVA and ANOSIM to add stats and see if controls are separate from samples
leg_neg_dm=distance.DistanceMatrix(square_bc, ids=sg_leg_neg["sample-id"])
pcoa_leg_neg_meta.index=pcoa_leg_neg_meta["sample-id"]
pcoa_leg_neg_meta["plottype"]=np.where(pcoa_leg_neg_meta["source"]=="leg", "sample", "control")
permanovaresults=distance.permanova(leg_neg_dm, pcoa_leg_neg_meta["plottype"])
anosimresults=distance.anosim(leg_neg_dm, pcoa_leg_neg_meta["plottype"])

#%%Supp Fig 6 D: comparing genus abundance in shotgun metagenomics and qPCR datasets

# using genus collapsed data to account for small primer specificity or classification differences at species level
#need a list with all genera colored, not just high abundance genera
allgeneracolor=pd.read_csv(work_dir+"/data/hex_color_keys/genus_color_chart.csv")
allgenuspal=dict(zip(allgeneracolor["taxa"], allgeneracolor["color"]))

# metagenomics: sggenus_rel_meta
# qpcr: pmpgenus_rel_meta

# plotting those genera shared between the two analysis methods
sg_pmp=list(set(sggenus_rel_meta.columns[:-16])&set(pmpgenus_rel_meta.columns[:-16]))
#powersoil only to compare to metagenomics
#leg only as paper focuses on low biomass sites and need to split by location to get accurate ave rel abundance

leg_pmp_sg=[]
genera=[]
foldchange=[]
sg_genus_ave=[]
for g in sg_pmp:
    sg_leg=sggenus_rel_meta[sggenus_rel_meta["source"]=="leg"][g].dropna()
    pmp_leg=pmpgenus_rel_meta[(pmpgenus_rel_meta["method"]=="powersoil")&(pmpgenus_rel_meta["source"]=="leg")][g].dropna()
    u2,p2=stats.mannwhitneyu(sg_leg,pmp_leg)
    sg_leg_ave=sg_leg.mean()
    fc=pd.Series(pmp_leg.mean()).div(sg_leg.mean()).values[0]
    leg_pmp_sg.append(p2)
    genera.append(g)
    foldchange.append(fc)
    sg_genus_ave.append(sg_leg_ave)
mannwhitney_pmp_sg_df=pd.DataFrame({"genera":genera, "leg_pval":leg_pmp_sg,"fc_pmp_over_sg":foldchange, "sg_ave":sg_genus_ave})

#plotting data as volcano plot to visualize fold change and significance comparing methods
#want to display genera in order of abundance in sg dataset
plt.rcParams["figure.figsize"] = (5,4)
pmp_sg_plot=mannwhitney_pmp_sg_df.sort_values(by="sg_ave", ascending=False)
sns.scatterplot(data=pmp_sg_plot, x=np.log2(pmp_sg_plot["fc_pmp_over_sg"]), y=-np.log10(pmp_sg_plot["leg_pval"]), hue="genera", palette=allgenuspal, linewidth=0.8, edgecolor="gray")
plt.ylim(0,10)
#plt.xlabel("Fold Change (log2)")
#plt.ylabel("-log10(P-value)")
plt.xlabel("")
plt.ylabel("")
#plt.title("qPCR vs shotgun metagenomics: ave abundance")
plt.hlines(2.51,-7,7, color="black", linestyle="dashed", alpha=0.7)
plt.vlines(1,0,10, color="black", linestyle="dashed", alpha=0.7)
plt.vlines(-1,0,10, color="black", linestyle="dashed", alpha=0.7)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xlim(-7,7)
plt.savefig(work_dir+"/supp_figures/FigS6D_sg_qpcr_volcano_leg_genus.svg", format="svg")
plt.show()

##literal comparison as a scatterplot

sg_ave=sggenus_rel_meta[sggenus_rel_meta["source"]=="leg"][sg_pmp].mean()
pmp_ave=pmpgenus_rel_meta[(pmpgenus_rel_meta["method"]=="powersoil")&(pmpgenus_rel_meta["source"]=="leg")][sg_pmp].mean()
sum_avedf=pd.DataFrame({"sg":sg_ave, "pmp":pmp_ave, "taxa":sg_pmp})
sum_avedf=sum_avedf.sort_values(by="sg", ascending=False)
sum_avedf["log_sg"]=np.log(sum_avedf["sg"]).replace([np.inf, -np.inf],0)
sum_avedf["log_pmp"]=np.log(sum_avedf["pmp"]).replace([np.inf, -np.inf],0)
sns.lmplot(data=sum_avedf, x="log_sg", y="log_pmp", scatter=False, ci=None, line_kws={'color': 'darkgray', 'linestyle': 'dashed'})
sns.scatterplot(data=sum_avedf, x="log_sg", y="log_pmp", hue="taxa", palette=allgenuspal, edgecolor="gray")
#plt.ylabel("Average rel abundance PMP qPCR")
plt.ylabel("")
plt.xlabel("")
#plt.xlabel("Average rel abundance shotgun metagenomics")
#plt.title("Genus abundance: PMP qPCR vs shotgun metagenomics")
plt.legend('',frameon=False)
#plt.yscale("log")
#plt.xscale("log")
plt.ylim(-8,0)
plt.xlim(-8,0)
plt.savefig(work_dir+"/supp_figures/FigS6D_sg_qpcr_scatterplot_leg_genus.png", format="png")
plt.show()
leg_genus_pearson=stats.pearsonr(sum_avedf["sg"], sum_avedf["pmp"])
print(leg_genus_pearson)

#%%Supplemental Fig 8A-B: unfiltered qPCR visualization of negative controls and mocks across extractions

#convert to relative abundance for plotting mock community composition
data=qPCR_skin_unfilt_abs.iloc[:,2:]
data["total"]=data.sum(axis=1)
data_rel=data.iloc[:,:-1].div(data["total"], axis=0)
print(data_rel.sum(axis=1)) #sanity check samples sum to 1 or 0 if they had no signal
# #rename df
qpcr_skin_rel=data_rel
qpcr_skin_rel["sample-id"]=qPCR_skin_unfilt_abs["sample-id"]
qpcr_skin_rel_meta=qpcr_skin_rel.merge(meta, how="inner", on="sample-id")

#add metadata to unfiltered absolute abundance df
qpcr_skin_unfilt_meta=qPCR_skin_unfilt_abs.merge(meta, how="inner", on="sample-id")

#Absolute abundance visualization to compare blanks from different extraction kits
absblank=qpcr_skin_unfilt_meta[(qpcr_skin_unfilt_meta["sampletype"]=="buffer")|(qpcr_skin_unfilt_meta["sampletype"]=="air")]
blanktotal=absblank.iloc[:,2:-16].sum(axis=0)
abovezero=blanktotal[blanktotal>0].index
absblank.index=absblank["sample-id"]
absblank=absblank.reindex([97,124,95,96,1,2]) #sort by desired kit order and add blank rows for zymo negative controls with no signal
absblank=absblank.fillna(0)

    ###Fig S8A: unfiltered blanks qPCR ###
plt.rcParams["figure.figsize"] = (3,3)
ax=absblank[abovezero].plot.bar(stacked=True, color=taxa_pal_sg_pmp, width=0.8, edgecolor="gray")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.ylim(0,2500)
plt.xticks([])
plt.xlabel("")
plt.ylabel("")
plt.savefig(work_dir+"/supp_figures/FigS8A_unfilt_qpcr_abs_blanks.svg", format="svg")
plt.show()

# relative abundance to compare mocks across different extraction kits #

lm_mock=qpcr_skin_rel_meta[qpcr_skin_rel_meta["sampletype"]=="lm_mock"].sort_values(by=["method", "source"])
lm_mock=lm_mock[lm_mock["method"]!="powersoil_tubes"]
#consistent viz across methods and figures
coltotal2=lm_mock.iloc[:,:-16].sum(axis=0)

abovezero2=coltotal2[coltotal2>0].index
#add back metadata to subset and reorder
includecol2=pd.concat([pd.Series(abovezero2), pd.Series(lm_mock.columns[-16:])])
#subset and concat for plotting 
lm_mockplot=lm_mock[includecol2]

mock_ps=lm_mockplot[(lm_mockplot["sampletype"]=="lm_mock")&(lm_mockplot["method"]=="powersoil")].iloc[:,:-16]
mock_rl=lm_mockplot[(lm_mockplot["sampletype"]=="lm_mock")&(lm_mockplot["method"]=="readylyse_dil")].iloc[:,:-16]
mock_zy=lm_mockplot[(lm_mockplot["sampletype"]=="lm_mock")&(lm_mockplot["method"]=="zymobiomics")].iloc[:,:-16]

allmethod_mock=pd.concat([mock_ps,mock_zy,mock_rl]).fillna(0)

#add other so that all bars appear
allmethod_mock["other"]=1-allmethod_mock.sum(axis=1)
#add spacer column to arrange bars
plt.rcParams["figure.figsize"] = (5,3.5)
allmethod_mock["spacer"]=[1,2,3,5,6,7,8,10,11,12,13]
spacelist=np.arange(1,14)
allmethod_mock=allmethod_mock.merge(pd.DataFrame({"spacer":spacelist}), on="spacer", how="right")
allmethod_mock.loc['ave']=allmethod_mock.iloc[:,:-1].mean(axis=0)
allmethod_mock=allmethod_mock.sort_values(by="ave", axis=1, ascending=False).iloc[:-1,:]
ax=allmethod_mock.iloc[:,:-1].plot.bar(stacked=True, color=taxa_pal_sg_pmp, width=0.9, linewidth=1, edgecolor="gray")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.ylim(0,1)
plt.ylabel("")
plt.xticks([])
plt.savefig(work_dir+"/supp_figures/FigS8A_unfilt_qpcr_relabun_mocks.svg", format="svg")
plt.show()
 #%%Supplemental Fig 8C-d: unfiltered 16S visualization of negative controls and mocks across extractions
#amplicon_unfilt=pd.read_csv("/Users/liebermanlab/MIT Dropbox/Laura Markey/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/2024_03_sample_processing/filter_fig_df/genus_16S_unfiltered.csv")
#remove row of duplicate (redid library prep) samples
amplicon_unfilt=amplicon_unfilt.iloc[:-13,:]
#fix column names of 16S dataset
unfilt_amp_names=amplicon_unfilt.columns[:-1]
unfilt_amp_name_rev=list(unfilt_amp_names.str[3:])
unfilt_name_final=unfilt_amp_name_rev+["sample-id"]
unfilt_name_final=pd.Series(unfilt_name_final).replace("Escherichia-Shigella", "Escherichia")
unfilt_16S=amplicon_unfilt
unfilt_16S.columns=unfilt_name_final
#remove Finegoldia
unfilt_16S_nof=unfilt_16S.drop(columns="Finegoldia")
unfilt_16S_nof["total"]=unfilt_16S_nof.iloc[:,:-1].sum(axis=1)
unfilt_16S_nof_renorm=unfilt_16S_nof.iloc[:,:-2].div(unfilt_16S_nof["total"],axis=0)
unfilt_16S_nof_renorm["sample-id"]=unfilt_16S["sample-id"].astype("int64")

##subset out and visualize blanks
#add genera to palette for plotting top blanks
blankhex=pd.read_csv(work_dir+"/data/hex_color_keys/genus_blank_hex.csv")
genusblank_pal=dict(zip(blankhex["taxa"], blankhex["color"]))
#add metadata
unfilt16s_meta=unfilt_16S_nof_renorm.merge(meta, how="inner", on="sample-id")
blanks=unfilt16s_meta[(unfilt16s_meta["sampletype"]=="buffer")|(unfilt16s_meta["sampletype"]=="air")]
blanktotal=blanks.iloc[:,:-16].sum(axis=0)
abovezero=blanktotal[blanktotal>0].index
#only including blanks processed alongside samples
dropdates=['20240711', '20240712']
blankplot=blanks[~blanks["date_processed"].isin(dropdates)]
    ### Fig. S8C (left)
plt.rcParams["figure.figsize"] = (4,4)
blankplot.index=blankplot["sample-id"]
blankplot=blankplot.reindex([97,124,96,95,1,2]) #extremely dumb solution manually setting order to get kits in desired order
ax=blankplot[abovezero].plot.bar(stacked=True, color=genusblank_pal, width=0.8, edgecolor="gray")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xticks([])
plt.xlabel("")
plt.ylim(0,1)
#plt.ylabel("Rel abundance")
plt.ylabel("")
#plt.title("Negative controls across methods")
plt.savefig(work_dir+"/supp_figures/FigS8C_amplicon_neg_unfilt_barplot.svg", format="svg")
plt.show()

#remove cutibacterium and renomalize so you can see everything else
blank_nocuti=blanks[abovezero].drop(columns="Cutibacterium")
blank_nocuti["total"]=blank_nocuti.sum(axis=1)
blank_noc_renorm=blank_nocuti.iloc[:,:-1].div(blank_nocuti["total"], axis=0)
blank_noc_renorm["sample-id"]=blanks["sample-id"].astype('int64')
blank_noc_meta=blank_noc_renorm.merge(meta, how="inner", on="sample-id")
blank_noc_plot=blank_noc_meta[~blank_noc_meta["date_processed"].isin(dropdates)]
blank_noc_plot.index=blank_noc_plot["sample-id"]
blank_noc_plot=blank_noc_plot.reindex([97,124,96,95,1,2]) #extremely dumb solution manually setting order to get kits in desired order
    ###FigS8C (right)
ax=blank_noc_plot.iloc[:,:-17].plot.bar(stacked=True, color=genusblank_pal, width=0.8, edgecolor="gray")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xticks([])
plt.ylim(0,1)
plt.xlabel("")
#plt.ylabel("Rel abundance")
plt.ylabel("")
#plt.title("Negative controls across methods")
plt.savefig(work_dir+"/supp_figures/FigS8C_amplicon_neg_unfilt_no_Cuti_barplot.svg", format="svg")
plt.show()

#subset out and visualize mock communities
mock_16s_unfilt=unfilt16s_meta[unfilt16s_meta["sampletype"]=="lm_mock"]
#remove weird powersoil tubes samples
mock_16s_unfilt=mock_16s_unfilt[mock_16s_unfilt["method"]!="powersoil_tubes"]
#manually reorder by sample-id 
mock_16s_unfilt.index=mock_16s_unfilt["sample-id"]
mock_16s_unfilt=mock_16s_unfilt.reindex([128,129,130,91,92,93,94,45,46,47,48])
coltotal=mock_16s_unfilt.iloc[:,:-16].sum(axis=0)
#all taxa detected in mocks
abovezero=coltotal[coltotal>0].index
#noninput only
inputlm=["Cutibacterium", "Staphylococcus", "Corynebacterium", "Escherichia", "Sporosarcina"]
noninput_mocks=abovezero[~abovezero.isin(inputlm)]
#including for visualization with all taxa
includecol=coltotal[coltotal>0.01].index
mock16s_plot=mock_16s_unfilt[includecol]
#add metadata for peace of mind and spacing
#plotting all three methods
plt.rcParams["figure.figsize"] = (4,3)
mock16s_plot["spacer"]=[1,2,3,5,6,7,8,10,11,12,13]
spacelist=np.arange(1,14)
mock16s_plot2=mock16s_plot.merge(pd.DataFrame({"spacer":spacelist}), on="spacer", how="right")
#setting order to plot genera
genusorder=["Cutibacterium", "Staphylococcus", "Corynebacterium", "Escherichia", "Streptococcus", "Enterobacteriaceae"]
    ###Fig. S8D (left)
ax=mock16s_plot2[genusorder].plot.bar(stacked=True, color=genusblank_pal, width=0.9, linewidth=1, edgecolor="gray")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.ylim(0,1)
plt.xticks([])
plt.savefig(work_dir+"/supp_figures/FigS8D_amplicon_mocks_unfilt_barplot.svg", format="svg")
plt.show()

#visualizing just the non-input species renormalized to 1
noninput_mockdata=mock_16s_unfilt.drop(columns=inputlm).reset_index(drop=True)
noninput_mockdata["total"]=noninput_mockdata.iloc[:,:-16].sum(axis=1)
noninput_mock_renorm=noninput_mockdata.iloc[:,:-17].div(noninput_mockdata["total"], axis=0)
noninput_mock_plot=noninput_mock_renorm[noninput_mocks]
noninput_mock_plot["spacer"]=[1,2,3,5,6,7,8,10,11,12,13]
noninput_mock_plot=noninput_mock_plot.merge(pd.DataFrame({"spacer":spacelist}), on="spacer", how="right")
noninput_mock_plot.loc["ave"]=noninput_mock_plot.mean()
noninput_mock_plot=noninput_mock_plot.sort_values(by="ave", axis=1, ascending=False)
    ###Fig. S8D right
ax=noninput_mock_plot[noninput_mocks].plot.bar(stacked=True, color=genusblank_pal, width=0.9, linewidth=1, edgecolor="gray")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.ylim(0,1)
plt.xticks([])
plt.savefig(work_dir+"/supp_figures/FigS8D_amplicon_mocks_unfilt_noninput_barplot.svg", format="svg")
plt.show()
#%%Supplemental table 1: are any genera associated with an extraction method?
# dataset is 16S for untargeted analysis: amplicon_nof_meta
# compare ps to zy; zy to rl; ps to rl
amplicon_compare=amplicon_nof_meta[(amplicon_nof_meta["source"]=="forehead")|(amplicon_nof_meta["source"]=="leg")]

ps_zy=[]
zy_rl=[]
ps_rl=[]
genera=[]
for g in amplicon_nof_renorm.columns[:-2]:
    ps=amplicon_compare[amplicon_compare["method"]=="powersoil"][g].dropna()
    rl=amplicon_compare[amplicon_compare["method"]=="readylyse_dil"][g].dropna()
    zy=amplicon_compare[amplicon_compare["method"]=="zymobiomics"][g].dropna()
    u1,p1=stats.mannwhitneyu(ps,rl)
    u2,p2=stats.mannwhitneyu(zy,rl)
    u3,p3=stats.mannwhitneyu(ps,zy)
    ps_zy.append(p3)
    zy_rl.append(p2)
    ps_rl.append(p1)
    genera.append(g)
mannwhitney_16S_genera_methods_df=pd.DataFrame({"genera":genera, "ps_zy":ps_zy,"zy_rl":zy_rl,"ps_rl":ps_rl})
mannwhitney_16S_genera_methods_df.to_csv(work_dir+"/main_supp_tables/Supplemental_table_X_compare_extraction_methods_genera_16s.csv")

# qPCR is more sensitive but can only compare across 3 methods for forehead: pmpgenus_rel_meta
# compare ps to zy; zy to rl; ps to rl
qpcrcompare_method=pmpgenus_rel_meta[pmpgenus_rel_meta["source"]=="forehead"]
ps_zy2=[]
zy_rl2=[]
ps_rl2=[]
genera2=[]
for g in pmpgenus_rel_meta.columns[:-16]:
    ps=qpcrcompare_method[qpcrcompare_method["method"]=="powersoil"][g].dropna()
    rl=qpcrcompare_method[qpcrcompare_method["method"]=="readylyse_dil"][g].dropna()
    zy=qpcrcompare_method[qpcrcompare_method["method"]=="zymobiomics"][g].dropna()
    u1,p1=stats.mannwhitneyu(ps,rl)
    u2,p2=stats.mannwhitneyu(zy,rl)
    u3,p3=stats.mannwhitneyu(ps,zy)
    ps_zy2.append(p3)
    zy_rl2.append(p2)
    ps_rl2.append(p1)
    genera2.append(g)
mannwhitney_qpcr_genera_methods_df=pd.DataFrame({"genera":genera2, "ps_zy":ps_zy2,"zy_rl":zy_rl2,"ps_rl":ps_rl2})
mannwhitney_qpcr_genera_methods_df.to_csv(work_dir+"/main_supp_tables/Supplemental_table_X_compare_extraction_methods_genera_forehead_qpcr.csv")


#%%Fig S9B: qPCR absolute abundance of FGT microbes 

#read in list of fgt targets to subset filtered qPCR absolute abundance dataset
biome_fgtlist=pd.read_csv(work_dir+"/data/raw_data/panel_qpcr/biome_fgt_species.csv", header=None)
#absolute abundance taxa and sample-filtered dataset: nofm_qPCR_absabun
fgtindataset=biome_fgtlist[0][biome_fgtlist[0].isin(nofm_qPCR_absabun.columns)]
qpcr_fgt=nofm_qPCR_absabun[fgtindataset] #slice based on fgt target
qpcr_fgt["sample-id"]=nofm_qPCR_absabun["sample-id"]
qpcr_fgt_meta=qpcr_fgt.merge(meta, how="inner", on="sample-id")

#subset out only powersoil samples (only sample with full FGT panel results)
ps_qpcr_fgt=qpcr_fgt_meta[qpcr_fgt_meta["method"]=="powersoil"]
#add plotting column to split out 4 subjects with fgt by metagenomics and other samples 
ps_qpcr_fgt["fgtleg"]=np.where(ps_qpcr_fgt["subject"].isin([3,5,11,17]), "fgtleg", "other")
#adding total column to see if we can visualize overall transfer from FGT this way : eh it's really just Gardnerella signal
ps_qpcr_fgt["fgtsum"]=ps_qpcr_fgt.iloc[:,:-17].sum(axis=1)
# sns.stripplot(data=ps_qpcr_fgt[ps_qpcr_fgt["sampletype"]=="swab"], x="fgtleg", y="fgtsum ", hue="source", palette=source_pal)
# plt.legend(bbox_to_anchor=(1.0,1.0))
# plt.yscale('log')
# plt.show()

#what fgt microbes are above zero? plot those in barplot?
fgtcolor=pd.read_csv(work_dir+"/data/hex_color_keys/fgt_colorkey.csv")
fgt_pal=dict(zip(fgtcolor["taxa"], fgtcolor["color"]))

#leg samples absolute abundance of FGT mcirobes

leg_fgt_qpcr=ps_qpcr_fgt[ps_qpcr_fgt["source"]=="leg"]
#include any taxa above zero
taxatotal=leg_fgt_qpcr.iloc[:,:-18].sum(axis=0)
abovezero=taxatotal[taxatotal>0].index
#reorder to match sg stacked barplots
leg_fgt_qpcr=leg_fgt_qpcr.set_index('subject').reindex(manorder_legs).reset_index() #set to same order as sg barplots
#only plotting species above zero
plot_leg_fgt=leg_fgt_qpcr[abovezero]
#sort by average abundance for ease of reading legend
plot_leg_fgt.loc["ave"]=plot_leg_fgt.mean()
plot_leg_fgt=plot_leg_fgt.sort_values(by="ave", axis=1, ascending=False)
fig, axs =plt.subplots()
plot_leg_fgt.iloc[:-1,:].plot.bar(ax=axs, stacked=True,width=0.8, linewidth=1, color=fgt_pal, edgecolor="black")
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xticks([])
axs.spines[['top','right']].set_visible(False)
plt.yscale('log')
plt.savefig(work_dir+"/supp_figures/FigS10B_leg_fgt_panel_abs_abundance.svg", format="svg")
plt.show()

#ofrehead samples aboslute abundance of FGT microbes
#this plot not included as figure but visualization included for innterest here #

fore_fgt_qpcr=ps_qpcr_fgt[ps_qpcr_fgt["source"]=="forehead"]
#include any taxa above zero
taxatotalf=fore_fgt_qpcr.iloc[:,:-18].sum(axis=0)
abovezerof=taxatotalf[taxatotalf>0].index
#reorder to match sg stacked barplots
fore_fgt_qpcr=fore_fgt_qpcr.set_index('subject').reindex(manorder_fore).reset_index() #set to same order as sg barplots
#only plotting species above zero
plot_fore_fgt=fore_fgt_qpcr[abovezero]
#sort by average abundance for ease of reading legend
plot_fore_fgt.loc["ave"]=plot_fore_fgt.mean()
plot_fore_fgt=plot_fore_fgt.sort_values(by="ave", axis=1, ascending=False)
fig, axs =plt.subplots()
plot_fore_fgt.iloc[:-1,:].plot.bar(ax=axs, stacked=True,width=0.8, linewidth=1, color=fgt_pal, edgecolor="black")
plt.legend(bbox_to_anchor=(1.0,1.0))
plt.xticks([])
axs.spines[['top','right']].set_visible(False)
plt.yscale('log')
plt.show()



