#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:47:03 2025

@author: liebermanlab
"""

#filtering datasets using a threshold based on the mock community non-input species
#done separately on each dataset: unique combination of analysis method and extraction method for qPCR and 16S 
#for each dataset visualize rel abundance of top 10 taxa in blanks, "set threshold" graph for non-input species >0, fration retained at various thresholds

#start from raw output of each analysis method
# directory of bracken files = shotgun metagenomics
# asv table = 16S; subset each extraction method
# absolute abundance table = qPCR; subset each extraction method

#output of this script is 3 figures / dataset x 7 datasets
#also output filtered datasets

#%%set up environment

#define scripts directory to import helper fucntions #
scripts_dir="/Users/liebermanlab/MIT Dropbox/Laura Markey/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/upload_code_data/scripts"
#define working directory for analysis #

work_dir="/Users/liebermanlab/MIT Dropbox/Laura Markey/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/upload_code_data"

import os

os.chdir(scripts_dir)

import pandas as pd
from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
import helper_functions as hf
import seaborn as sns
import matplotlib.patches as mpatches

#os.chdir(work_dir)

#read in metadata which is used for all samples (all methods)

meta=pd.read_csv(work_dir+"/data/samplemetadata.csv")
meta["sample-id"]=meta["sample-id"].astype("str")

#read in color palette used throughout for consistency
taxacolor_species=pd.read_csv(work_dir+"/data/hex_color_keys/rev4_with_top_blanks_hex.csv")
species_pal=dict(zip(taxacolor_species["taxa"], taxacolor_species["color"]))

#%%METAGENOMICS ABSOLUTE ABUNDANCE TO GET MICROBIAL READS FOR MICRO-HUMAN READ RATIO ANALYSIS AND QC ###
#go to dir that holds bracken output
os.chdir(work_dir+"/data/raw_data/metagenomics_sequencing/kraken2_bracken_output")
dir=os.getcwd()
#combine all bracken files listed in dfs into a single csv based on value of the taxonomy_id column
dfs2=[]
for filename in os.listdir(dir):
    if filename.endswith(".bracken"):
        og=pd.read_table(filename)
        taxa=og["name"]
        sample=str(filename)
        sample_df=pd.DataFrame({"taxa":taxa, str(sample):og["new_est_reads"]})
        dfs2.append(sample_df)
        continue
    else:
        continue
#combine into one df
merged2=reduce(lambda left,right: pd.merge(left,right,on=['taxa'],how='outer'),dfs2).fillna('0')
#flip so that samples are rows and taxa are columns
merged_flip2=merged2.T
taxa_col2=merged_flip2.iloc[0,:]
#subset out to rename columns
abs_abun=merged_flip2.iloc[1:,:]
abs_abun.columns=taxa_col2
#make metadata columns
#grab sample name
sample_nob2=pd.Series(abs_abun.index).str.split("_Human.filt9606.bracken", expand=True)[0]
#pull out sample id
sample_id2=sample_nob2.str[-3:]
#create condition and choice list to identify dilution of beads used based on location 
cond_list=[sample_nob2.str.contains("forehead"), sample_nob2.str.contains("leg"), sample_nob2.str.contains("mock"), sample_nob2.str.contains("buffer"), sample_nob2.str.contains("air")]
choice=[sample_nob2.str[8:10], sample_nob2.str[3:5], sample_nob2.str[5:7], sample_nob2.str[6:8], sample_nob2.str[3:5]]
choice2=["forehead", "leg", "mock", "buffer", "air"]
sample_dil=np.select(cond_list, choice, "None")
sample_source=np.select(cond_list, choice2, "other")
#combine with relabun to plot with metadata
abs_abun=abs_abun.reset_index()
abs_abun["sample-id"]=sample_id2
abs_abun["tag_bead_dil"]=sample_dil
abs_abun["source"]=sample_source
abs_abun["sample-id"]=abs_abun["sample-id"].replace("097", "97")

#getting just bacteria reads to compare to 16S copy number

sg_fungus=abs_abun.columns[(abs_abun.columns.str.contains("Malassezia"))|(abs_abun.columns.str.contains("Candida"))|(abs_abun.columns.str.contains("Aspergillus"))]
sg_virus=abs_abun.columns[abs_abun.columns.str.contains("virus")]
sg_discard=pd.concat([pd.Series(sg_fungus),pd.Series(sg_virus)])

abs_justbacteria=abs_abun.drop(columns=sg_discard)
#add column total non-viral reads
abs_justbacteria["microbe_reads"]=abs_justbacteria.iloc[:,1:-3].astype("int64").sum(axis=1)
#remove 1:20 tag bead samples
abs_justbacteria=abs_justbacteria[abs_justbacteria["tag_bead_dil"]=="50"]
#subset out just the total microbial reads columns
totalmicroreads=abs_justbacteria[["sample-id", "microbe_reads"]]
totalmicroreads["sample-id"]=totalmicroreads["sample-id"].astype("int64")
#save to csv - this will be combined with human reads and metadata in big analysis/viz script
totalmicroreads.to_csv(work_dir+"/data/output_read_filter_datasets/bracken_micro_reads.csv", index=None)

#%% ## FILTER SHOTGUN METAGENOMICS - COMBINE UNFILTERED BRACKEN OUTPUT, VISUALIZATION RAW DATA, GENERATE FILTERED DATASET ####
#### read in and combine unfiltered bracken data ##

#read in raw bracken output
#go to dir that holds bracken output
os.chdir(work_dir+"/data/raw_data/metagenomics_sequencing/kraken2_bracken_output")
dir=os.getcwd()

#combine all bracken files (now dataframes) listed in dfs into a single csv based on value of the taxonomy_id column
dfs=[]
for filename in os.listdir(dir):
    if filename.endswith(".bracken"):
        og=pd.read_table(filename)
        taxa=og["name"]
        sample=str(filename)
        sample_df=pd.DataFrame({"taxa":taxa, str(sample):og["fraction_total_reads"]})
        dfs.append(sample_df)
        continue
    else:
        continue
#combine into one df
merged=reduce(lambda left,right: pd.merge(left,right,on=['taxa'],how='outer'),dfs).fillna('0')

print("Number of samples included:",len(dfs))

## format and save df
#os.chdir('/Users/liebermanlab/MIT Dropbox/Laura Markey/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/2024_03_sample_processing/')#change to meta dir

#flip so that samples are rows and taxa are columns
merged_flip=merged.T
taxa_col=merged_flip.iloc[0,:]
#subset out to rename columns
relabun=merged_flip.iloc[1:,:]
relabun.columns=taxa_col
#make metadata columns
#grab sample name
sample_nob=pd.Series(relabun.index).str.split("_Human.filt9606.bracken", expand=True)[0]
#pull out sample id
sample_id=sample_nob.str[-3:]
#create condition and choice list to identify dilution of beads used based on location 
cond_list=[sample_nob.str.contains("forehead"), sample_nob.str.contains("leg"), sample_nob.str.contains("mock"), sample_nob.str.contains("buffer"), sample_nob.str.contains("air")]
choice=[sample_nob.str[8:10], sample_nob.str[3:5], sample_nob.str[5:7], sample_nob.str[6:8], sample_nob.str[3:5]]
choice2=["forehead", "leg", "mock", "buffer", "air"]
sample_dil=np.select(cond_list, choice, "None")
sample_source=np.select(cond_list, choice2, "other")
#combine with relabun to plot with metadata
relabun=relabun.reset_index()
relabun["sample-id"]=sample_id
relabun["tag_bead_dil"]=sample_dil
relabun["sample-id"]=relabun["sample-id"].replace("097", "97")

#subset out only 1 dilution of beads (no repeated samples)
#only analyzing samples tagmented with 1:50 dilution of beads
relabun=relabun[relabun["tag_bead_dil"]=="50"]

#drop the bead dilution column to be comparable to other datasets: only non sample data is sample-id
relabun=relabun.drop(columns="tag_bead_dil")
#add metadata to relabun
relabun_meta=relabun.merge(meta, how="inner", on="sample-id")


#### remove samples below read filter so that bad samples are not used to choose taxa threshold ###
totalmicroreads["sample-id"]=totalmicroreads["sample-id"].astype("str")
microreads_meta=totalmicroreads.merge(meta, how="inner", on="sample-id")
#micro reads histogram 
histdata2=microreads_meta[microreads_meta["sampletype"]=="swab"]["microbe_reads"]
binwidth2=50000
plt.rcParams["figure.figsize"] = (5,3)
plt.hist(histdata2, bins=np.arange(0,6e6, binwidth2),color="#b2849a", edgecolor="black", linewidth=0.5)
plt.vlines(1e5, 0,10, color="black", linestyle="dashed",linewidth=0.8) #100,000 is pretty low for metagenomics
plt.ylim(0,5)
plt.xlabel("Microbial Reads (bracken)")
plt.ylabel("Sample counts")
#plt.xscale('log')
plt.savefig(work_dir+"/data/output_read_filter_datasets/microbe_reads_histogram.png",format="png")
plt.show()

drop_sg_samples=microreads_meta[(microreads_meta["sampletype"]=="swab")&(microreads_meta["microbe_reads"]<=100000)]["sample-id"]
drop_sg_samples.to_csv(work_dir+"/data/output_read_filter_datasets/dropsg_lowreads.csv")

#remove bad samples from both datasets - we use relabunmeta to filter taxa and then save relabun for future analysis and viz

relabun_meta=relabun_meta[~relabun_meta["sample-id"].isin(drop_sg_samples)]

relabun=relabun[~relabun["sample-id"].isin(drop_sg_samples)]

#### Visualize unfiltered data average source composition ###
# Fig. 2A
#split by site and convert to numeric
leg_numeric=relabun_meta[relabun_meta["source"]=="leg"].iloc[:,1:-16].apply(pd.to_numeric).reset_index(drop=True)
fore_numeric=relabun_meta[relabun_meta["source"]=="forehead"].iloc[:,1:-16].apply(pd.to_numeric).reset_index(drop=True)
blank_numeric=relabun_meta[(relabun_meta["source"]=="air")|(relabun_meta["source"]=="buffer")].iloc[:,1:-16].apply(pd.to_numeric).reset_index(drop=True)
mock_numeric=relabun_meta[relabun_meta["sampletype"]=="lm_mock"].iloc[:,1:-16].apply(pd.to_numeric)
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
mockinput=pd.Series(["Cutibacterium acnes", "Staphylococcus aureus", "Staphylococcus epidermidis", "Corynebacterium accolens", "Escherichia coli"])
leg_numeric.loc["ave"]=leg_numeric.mean(axis=0)
topleg=pd.Series(leg_numeric.sort_values(by="ave", axis=1, ascending=False).columns[:20])
#include both top 10 blank and mock inputs in barplot
plotme_unfilt=pd.Series(pd.concat([mockinput,topblank, topleg]).unique())
#make small and skinny
#this is fig2a
#plt.rcParams["figure.figsize"] = (4,4)
fig,ax=plt.subplots()
blank_aveskin=pd.concat([blank_numeric.iloc[:-1,:],leg_fore_df,mock_ave])
blank_aveskin=blank_aveskin.fillna(0)
blank_aveskin=blank_aveskin.loc[:, (blank_aveskin != 0).any(axis=0)]
#add random unknown column to fill all samples to 1
blank_aveskin["other"]=1-blank_aveskin[plotme_unfilt].sum(axis=1)
#add other to list of columns to plot
finalplot=pd.concat([plotme_unfilt, pd.Series("other")])
ax=blank_aveskin[finalplot].plot.bar(stacked=True, color=species_pal, width=0.9, linewidth=1, edgecolor="black")
plt.xticks([0,1,2,3,4], ["Collection", "Extraction", "Leg", "Forehead","Mock"],rotation=15, fontsize=12)
legend=ax.legend(bbox_to_anchor=(1.0,1.0))
for text in legend.get_texts():
    text.set_fontstyle('italic')
plt.ylim(0,1)
#plt.ylabel("Rel abundance")
plt.ylabel("")
#plt.savefig(work_dir+"/main_figures/Fig2A_control_mock_unfilt.svg", format="svg")
plt.show()
#completely unfiltered bracken results

relabun.to_csv(work_dir+"/data/output_read_filter_datasets/unfilt_bracken_sg_meta.csv", index=False)

#%% still metagenomics: filtering and visualizing #####

#filter dataset and generate 2 additional figures to show how filter works ##

###finding threshold from mock samples ###

#input for the set threshold function: 
#slice only mock samples with and without metadata
mock_meta=relabun_meta[relabun_meta["sampletype"]=="lm_mock"]
mock_df=mock_meta.iloc[:,1:-16].apply(pd.to_numeric)
#list of species in mock community
inputspecies=["Cutibacterium acnes", "Staphylococcus epidermidis", "Staphylococcus aureus", "Corynebacterium accolens", "Escherichia coli"]
#use set_threshold helper function to generate 1 plot and the threshold to filter data
sg_df1,sg_fig, m3thresh=hf.set_threshold_rev(mock_meta, mock_df, inputspecies)
#Plot in Fig. 2B
sg_fig.get_figure().savefig(work_dir+"/main_figures/Fig2B_sg_filter_set_threshold.png", bbox_inches="tight")

#use perc_retain function to generate 1 plot
#this is the example shown in Fig2C
#input datasets for each source and the threshold to plot as vertical line
#subset data by source prior to using function because different datasets have different columns of metadata
legdata=relabun_meta[relabun_meta["source"]=="leg"].iloc[:,1:-16].apply(pd.to_numeric)
foredata=relabun_meta[relabun_meta["source"]=="forehead"].iloc[:,1:-16].apply(pd.to_numeric)
airdata=relabun_meta[relabun_meta["source"]=="air"].iloc[:,1:-16].apply(pd.to_numeric)
bufferdata=relabun_meta[relabun_meta["source"]=="buffer"].iloc[:,1:-16].apply(pd.to_numeric)
sg_perc_retain_fig,leg_retain, fore_retain=hf.perc_retain(legdata, foredata,bufferdata, airdata, m3thresh)
sg_perc_retain_fig.get_figure().savefig(work_dir+"/main_figures/Fig2C_sg_perc_retain_each_threshold.png", bbox_inches="tight")
plt.show()

#use threshold from above to filter data and return filtered and renormalized dataset

#just relative abundance unfiltered data and sample-id column (and first column which is index)
relabun.iloc[:,1:-1]=relabun.iloc[:,1:-1].apply(pd.to_numeric)
#replace values below threshold with 0
filt1_relabun=relabun.iloc[:,1:-1]
filt1_relabun[filt1_relabun<m3thresh]=0
#drop taxa that are not detected in any samples
taxasum=filt1_relabun.sum() #sum all taxa columns
dropme=taxasum[taxasum==0].index #make a list of columns not detected in any samples
filt2_relabun=filt1_relabun.drop(columns=dropme)
#RENORMALIZEE
filt2_relabun["total"]=filt2_relabun.sum(axis=1)
filt2_renorm=filt2_relabun.iloc[:,:-1].div(filt2_relabun["total"],axis=0)
#add back metadata
filt2_renorm["sample-id"]=relabun["sample-id"]
#save file
filt2_renorm.to_csv(work_dir+"/data/output_read_filter_datasets/sg_filtered_bracken.csv", index=False)

#%% #### visualizing unfiltered 16S dataset and filtering by threshold ####
#read in unfiltered output from QIIME2
#need to combine ASV table and taxonomy and then collapse to the genus level and flip so samples are rows

#read in starting data
#unmodified qiime2 output (DADA2 followed by bayes classifier trained on new SILVA database from VK)
unfilt_ASV=pd.read_csv(work_dir+"/data/raw_data/16s_sequencing/asv.tsv", sep="\t")
taxonomytable=pd.read_csv(work_dir+"/data/raw_data/16s_sequencing/taxonomy.tsv", sep="\t")

#create taxonomy lookup dataframe
taxonomytable=taxonomytable.rename(columns={"Feature ID": "ASV"})
taxa_exp=taxonomytable.Taxon.str.split(pat=";", expand=True).iloc[:,:7]
columns=["k", "p", "c", "o", "f", "g", "s"]
taxa_exp.columns=columns
taxa_exp["ASV"]=taxonomytable["ASV"]
phyloforwardtaxa=taxa_exp.fillna(method="ffill", axis=1)
#to prevent "g__uncultured" being the final taxonomic designation pull from f level #
phyloforwardtaxa["genus"]=np.where(phyloforwardtaxa["g"]=="g__uncultured", phyloforwardtaxa["f"], phyloforwardtaxa["g"])
#to eliminate "f__uncultured" do it again 
phyloforwardtaxa["genus"]=np.where(phyloforwardtaxa["genus"]=="f__uncultured", phyloforwardtaxa["o"], phyloforwardtaxa["genus"])

#transform ASV table and add sample-id information
t_asv=unfilt_ASV.T
#rename columns as ASVs
t_asv.columns=t_asv.iloc[0,:]
#drop first row which is ASVs
t_asv=t_asv.drop(index="#OTU ID")
#add sample columns
t_asv["sample-id"]=t_asv.index
#reset index to numeric
t_asv=t_asv.reset_index(drop=True)
#make asv dataframe numeric
t_asv.iloc[:,:-1]=t_asv.iloc[:,:-1].apply(pd.to_numeric)

#make an unfiltered ASV table that includes taxonomy labels 
unfilt_asv_taxa=t_asv.iloc[:,:-1]
labelrow1=[]
for c in unfilt_asv_taxa.columns:
    s=phyloforwardtaxa[phyloforwardtaxa["ASV"]==c]["genus"].values[0]
    labelrow1.append(s)
unfilt_asv_taxa.loc["genus"]=labelrow1
#unfilt_asv_taxa.columns=unfilt_asv_taxa.loc["genus"]
#flip it for ease of visualizing
flip_unfilt_taxa=unfilt_asv_taxa.T
columnlist=pd.concat([t_asv["sample-id"], pd.Series("genus")])
flip_unfilt_taxa.columns=columnlist
#write to csv as supplemental table
flip_unfilt_taxa.to_csv(work_dir+"/main_supp_tables/unfiltered_asv_table_taxa.csv")
## 

#add taxonoamy labels and collapse to the genus level
#lists of ASVs to exclude from analysis
undef=phyloforwardtaxa[phyloforwardtaxa["g"]=="Unassigned"]["ASV"]
bact=phyloforwardtaxa[phyloforwardtaxa["g"]=="d__Bacteria"]["ASV"]
euk=phyloforwardtaxa[phyloforwardtaxa["k"]=="d__Eukaryota"]["ASV"]

#remove these columns
asv_filt1=t_asv.drop(columns=undef) #remove undefined ASVs
asv_filt1a=asv_filt1.drop(columns=euk) #remove eukaryote ASVs
asv_filt2=asv_filt1a.drop(columns=bact) #remove ASVs not defined below the level of "Bacteria"

#sum all asvs included in analysis to get number of reads used for profiling
asv_assigned_total=pd.DataFrame({"sample-id":asv_filt2["sample-id"], "reads_assigned_taxa": asv_filt2.iloc[:,:-1].sum(axis=1)})
#asv_assigned_total.to_csv("/Users/liebermanlab/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/2024_03_sample_processing/total_16s_reads_assigned_asvs.csv")

#generate df for another QC figure: what perc of reads for 16S were assigned/not assigned for different sources?
undef_bactonly_asv=pd.DataFrame({"sample-id":t_asv["sample-id"], "undef_reads":t_asv[undef].sum(axis=1), "bact_reads":t_asv[bact].sum(axis=1), "euk_reads":t_asv[euk].sum(axis=1), "totalreads":t_asv.iloc[:, :-1].sum(axis=1)})
#remove duplicate pcr reactions at the end
asv_tax_distr_num=undef_bactonly_asv.iloc[:-12,:].astype('int64')
#combine undefined and poorly defined reads
asv_tax_distr_num["bad_class_reads"]=asv_tax_distr_num["undef_reads"]+asv_tax_distr_num["bact_reads"]
#write to csv to make the figure in the main analysis doc
asv_tax_distr_num.to_csv(work_dir+"/data/output_read_filter_datasets/16sqc_perc_undef_bad_class.csv")

####remove samples with too few reads from 16S dataset - filter bad samples prior to filtering bad taxa ####

# read in stats from dada2 #
dada2stats=pd.read_csv(work_dir+"/data/raw_data/16s_sequencing/stats.tsv", sep="\t").iloc[:-12,:]
dada2stats["sample-id"]=dada2stats["sample-id"].astype('int64')
# read in number of reads actually assigned to taxa from read_filter_datasets.py
asv_assigned_total=asv_assigned_total.iloc[:-12,:]
asv_assigned_total["sample-id"]=asv_assigned_total["sample-id"].astype('int64')
#combine 16s qc data 
asv_stats=dada2stats.merge(asv_assigned_total, how="inner", on="sample-id")
asv_stats["sample-id"]=asv_stats["sample-id"].astype("str")
#add metadata
asv_stats_meta=asv_stats.merge(meta, how="inner", on="sample-id")

#histogram to look at reads assigned taxa
histdata4=asv_stats_meta[asv_stats_meta["sampletype"]=="swab"]["reads_assigned_taxa"]
binwidth4=750
plt.rcParams["figure.figsize"] = (5,3)
plt.hist(histdata4, bins=np.arange(0,1e5, binwidth4),color="#b2849a", edgecolor="black", linewidth=0.5)
plt.vlines(500, 0,10, color="black", linestyle="dashed", linewidth=0.8)
plt.ylim(0,10)
plt.xlabel("16S reads assigned taxa")
plt.ylabel("Sample counts")
#plt.xscale('log')
plt.savefig(work_dir+"/supp_figures/FigS2B_amplicon_reads_assigned_taxa_histogram.png",format="png")
plt.show()

#16S filter reads #
drop_16s_samples=asv_stats_meta[(asv_stats_meta["sampletype"]=="swab")&(asv_stats_meta["reads_assigned_taxa"]<=500)]["sample-id"]
drop_16s_samples.to_csv(work_dir+"/data/output_read_filter_datasets/drop16s_lowreads.csv")

#collapse remaining asv table at genus level
asvcol=asv_filt2.iloc[:,:-1]
labelrow=[]
for c in asvcol.columns:
    s=phyloforwardtaxa[phyloforwardtaxa["ASV"]==c]["genus"].values[0]
    labelrow.append(s)
asvcol.loc["genus"]=labelrow
asvcol.columns=asvcol.loc["genus"]
#sum columns with the same genus or other phylogenetic designation name
genus_raw_coll=asvcol.groupby(axis=1,level=0).sum()
#convert to rel abundance
genus_raw_coll["total"]=genus_raw_coll.sum(axis=1)
genus_raw_coll=genus_raw_coll.drop(index="genus")
#assign sample-ids so you can make a df with correct mapping of id to row
genus_raw_coll["sample-id"]=asv_filt1["sample-id"]
#omit rows with a total of zero reads
genus_raw_nozero=genus_raw_coll[genus_raw_coll["total"]>0]
genus_rel=genus_raw_nozero.iloc[:,:-2].div(genus_raw_nozero["total"],axis=0)
genus_rel.loc["ave"]=genus_rel.mean()
genus_rel=genus_rel.sort_values(by="ave", axis=1, ascending=False)
#add sample column back 
genus_rel["sample-id"]=genus_raw_nozero["sample-id"]

#drop bad samples
genus_rel=genus_rel[~genus_rel["sample-id"].isin(drop_16s_samples)]

#save unfiltered dataset
genus_rel.to_csv(work_dir+"/data/output_read_filter_datasets/genus_16S_unfiltered.csv", index=False)

##### Supp Figure: unfiltered composition ###
#split by method
#use 16s function to avoid giant blocks of repeated code

#add metadata
genus_rel_meta=genus_rel.merge(meta, how="inner", on="sample-id")

#drop extra buffers from weird day that you tried to process the ATCC mock community
genus_rel_meta=genus_rel_meta[genus_rel_meta["date_processed"]!="20240711"]

#fix column names to remove prefix
newcolumns=pd.concat([pd.Series(genus_rel_meta.columns[:-16].str[3:]), pd.Series(genus_rel_meta.columns[-16:])])
newcolumns=newcolumns.replace("Escherichia-Shigella", "Escherichia")
newcolumns=newcolumns.replace("HT002", "Lactobacillus")
newcolumns=newcolumns.replace("Prevotella_7", "Prevotella")
genus_rel_meta.columns=newcolumns

#split by extraction method to plot individually
genus_ps=genus_rel_meta[genus_rel_meta["method"]=="powersoil"]
genus_rl=genus_rel_meta[genus_rel_meta["method"]=="readylyse_dil"]
genus_zy=genus_rel_meta[genus_rel_meta["method"]=="zymobiomics"]

#helper function to plot the unfiltered average abundance of top20 genera in blanks + mock community inpuots
# genus_color=pd.read_csv(work_dir+"/data/hex_color_keys/genus_color_chart.csv")
# genus_pal=dict(zip(genus_color["taxa"], genus_color["color"]))
# #powersoil
# genus_ps_unfilt=hf.viz_unfilt_16s(genus_ps, genus_pal)
# genus_ps_unfilt.get_figure().savefig(work_dir+"/supp_figures/FigSX_unfilt_16S_ps.png", bbox_inches="tight")
# #zymo
# genus_zymo_unfilt=hf.viz_unfilt_16s(genus_zy, genus_pal)
# genus_zymo_unfilt.get_figure().savefig(work_dir+"/supp_figures/FigSX_unfilt_16S_zy.png", bbox_inches="tight")
# #readylyse
# genus_rl_unfilt=hf.viz_unfilt_16s(genus_rl, genus_pal)
# genus_rl_unfilt.get_figure().savefig(work_dir+"/supp_figures/FigSX_unfilt_16S_rl.png", bbox_inches="tight")
#%%##### Supp Figure 3: find threshold and plot percent retained for 16S datasets; filter and save data
#filter dataset and generate 2 additional figures to show how filter works ##

### Powersoil ###

###finding threshold from mock samples ###
#input for the set threshold function: 
#slice only mock samples with and without metadata
genus_ps_mock=genus_ps[genus_ps["sampletype"]=="lm_mock"]
mock_df=genus_ps_mock.iloc[:,:-16].apply(pd.to_numeric)
#list of genera in mock community
inputspecies=["Cutibacterium", "Staphylococcus","Corynebacterium", "Escherichia"]

#use set_threshold helper function to generate 1 plot and the threshold to filter data
amp_ps_df,amp_ps_16s,ps_thresh=hf.set_threshold_rev(genus_ps_mock,mock_df,inputspecies)
amp_ps_16s.get_figure().savefig(work_dir+"/supp_figures/FigS4_amplicon_16s_powersoil_set_threshold.png",bbox_inches="tight")

#use perc_retain function to generate 1 plot
#input datasets for each source and the threshold to plot as vertical line
#subset data by source prior to using function because different datasets have different columns of metadata
legdata=genus_ps[genus_ps["source"]=="leg"].iloc[:,:-16].apply(pd.to_numeric)
foredata=genus_ps[genus_ps["source"]=="forehead"].iloc[:,:-16].apply(pd.to_numeric)
airdata=genus_ps[genus_ps["source"]=="air"].iloc[:,:-16].apply(pd.to_numeric)
bufferdata=genus_ps[genus_ps["source"]=="buffer"].iloc[:,:-16].apply(pd.to_numeric)
amp_ps_16s_perc_retain, ps_leg_retain, ps_fore_retain=hf.perc_retain(legdata, foredata,bufferdata, airdata, ps_thresh)
amp_ps_16s_perc_retain.get_figure().savefig(work_dir+"/supp_figures/FigS4_amplicon_16s_powersoil_perc_retain.png", bbox_inches="tight")

#use threshold from above to filter data and return filtered and renormalized dataset

#just relative abundance unfiltered data and sample-id column (and first column which is index)
genus_ps.iloc[:,:-16]=genus_ps.iloc[:,:-16].apply(pd.to_numeric)
#replace values below threshold with 0
filt1_genus_ps=genus_ps.iloc[:,:-16]
filt1_genus_ps[filt1_genus_ps<ps_thresh]=0

#### Zymobiomics ###

###finding threshold from mock samples ###
#input for the set threshold function: 
#slice only mock samples with and without metadata
genus_zy_mock=genus_zy[genus_zy["sampletype"]=="lm_mock"]
mock_df=genus_zy_mock.iloc[:,:-16].apply(pd.to_numeric)
#list of genera in mock community
inputspecies=["Cutibacterium", "Staphylococcus","Corynebacterium", "Escherichia"]

#use set_threshold helper function to generate 1 plot and the threshold to filter data
amp_zy_df, amp_zy_16s,zy_thresh=hf.set_threshold_rev(genus_zy_mock,mock_df,inputspecies)
amp_zy_16s.get_figure().savefig(work_dir+"/supp_figures/FigS4_amplicon_16s_zymobiomics_set_threshold.png",bbox_inches="tight")

#use perc_retain function to generate 1 plot
#input datasets for each source and the threshold to plot as vertical line
#subset data by source prior to using function because different datasets have different columns of metadata
legdata=genus_zy[genus_zy["source"]=="leg"].iloc[:,:-16].apply(pd.to_numeric)
foredata=genus_zy[genus_zy["source"]=="forehead"].iloc[:,:-16].apply(pd.to_numeric)
airdata=genus_zy[genus_zy["source"]=="air"].iloc[:,:-16].apply(pd.to_numeric)
bufferdata=genus_zy[genus_zy["source"]=="buffer"].iloc[:,:-16].apply(pd.to_numeric)
amp_zy_16s_perc_retain, zy_leg_retain, zy_fore_retain=hf.perc_retain(legdata, foredata,bufferdata, airdata, zy_thresh)
amp_zy_16s_perc_retain.get_figure().savefig(work_dir+"/supp_figures/FigS4_amplicon_16s_zymobiomics_perc_retain.png", bbox_inches="tight")

#use threshold from above to filter data and return filtered and renormalized dataset

#just relative abundance unfiltered data and sample-id column (and first column which is index)
genus_zy.iloc[:,:-16]=genus_zy.iloc[:,:-16].apply(pd.to_numeric)
#replace values below threshold with 0
filt1_genus_zy=genus_zy.iloc[:,:-16]
filt1_genus_zy[filt1_genus_zy<zy_thresh]=0

##### Ready Lyse ####
#readylyse is weird because it doesn't have all 4 input genera in m3; but now the threshold is based on non-input genera so I think it is fine? Also it's very comparable to zymo

###finding threshold from mock samples ###
#input for the set threshold function: 
#slice only mock samples with and without metadata
genus_rl_mock=genus_rl[genus_rl["sampletype"]=="lm_mock"]
mock_df=genus_rl_mock.iloc[:,:-16].apply(pd.to_numeric)
#list of genera in mock community
inputspecies=["Cutibacterium", "Staphylococcus","Corynebacterium", "Escherichia"]

#use set_threshold helper function to generate 1 plot and the threshold to filter data
amp_rl_df, amp_rl_16s,rl_thresh=hf.set_threshold_rev(genus_rl_mock,mock_df,inputspecies)
amp_rl_16s.get_figure().savefig(work_dir+"/supp_figures/FigS4_amplicon_16s_readylyse_set_threshold.png",bbox_inches="tight")

#use perc_retain function to generate 1 plot
#input datasets for each source and the threshold to plot as vertical line
#subset data by source prior to using function because different datasets have different columns of metadata
legdata=genus_rl[genus_rl["source"]=="leg"].iloc[:,:-16].apply(pd.to_numeric)
foredata=genus_rl[genus_rl["source"]=="forehead"].iloc[:,:-16].apply(pd.to_numeric)
airdata=genus_rl[genus_rl["source"]=="air"].iloc[:,:-16].apply(pd.to_numeric)
bufferdata=genus_rl[genus_rl["source"]=="buffer"].iloc[:,:-16].apply(pd.to_numeric)
amp_rl_16s_perc_retain, rl_leg_retain, rl_fore_retain=hf.perc_retain(legdata, foredata,bufferdata, airdata, rl_thresh)
amp_rl_16s_perc_retain.get_figure().savefig(work_dir+"/supp_figures/FigS4_amplicon_16s_readylyse_perc_retain.png", bbox_inches="tight")

#use threshold from above to filter data and return filtered and renormalized dataset

#just relative abundance unfiltered data and sample-id column (and first column which is index)
genus_rl.iloc[:,:-16]=genus_rl.iloc[:,:-16].apply(pd.to_numeric)
#replace values below threshold with 0
filt1_genus_rl=genus_rl.iloc[:,:-16]
filt1_genus_rl[filt1_genus_rl<rl_thresh]=0

### all extraction methods ###
#combine into a single dataframe of filtered data to analyze and compare to shotgun and qPCR data
genus_comb_filt=pd.concat([filt1_genus_ps,filt1_genus_zy, filt1_genus_rl])
comb_id=pd.concat([genus_ps["sample-id"], genus_zy["sample-id"], genus_rl["sample-id"]])
#drop taxa that are not above 0 in any samples
taxsum=genus_comb_filt.sum()
dropzero=pd.Series(taxsum[taxsum==0].index)
filtgenus_drop=genus_comb_filt.drop(columns=dropzero)
#renormalize
filtgenus_drop["total"]=filtgenus_drop.sum(axis=1)
filtgenus_renorm=filtgenus_drop.iloc[:,:-1].div(filtgenus_drop["total"], axis=0)
#add back metadata
filtgenus_renorm["sample-id"]=comb_id

#save file
filtgenus_renorm.to_csv(work_dir+"/data/output_read_filter_datasets/amp_16s_genus_filtered.csv", index=False)

#%% ##### qPCR data: need to read in and visualize the unfiltered data from each extraction method #####

#recall that PowerSoil samples were also run on the FGT panel, while readylyse and zymo samples were only run on the skin panel

#### will read in and combine datasets and then filter individually (on different # of taxa) ###
#two qPCR runs using skin panel only, all extraction methods
pmp_skin_data=pd.read_csv(work_dir+"/data/raw_data/panel_qpcr/skin_panel_full_samples.csv") # this file includes both runs
skinonlytaxa=pmp_skin_data.columns[2:] #first two columns are sample ID and universal 16S
#vaginal microbiome panel results (powersoil only)
pmp_fgt=pd.read_csv(work_dir+"/data/raw_data/panel_qpcr/vaginal_panel_ps_only.csv")
#when there are overlapping columns between the two panels we are using the value from the skin panel results
overlaptaxa=pmp_fgt.columns[pmp_fgt.columns.isin(pmp_skin_data.columns)]
overlapdrop=overlaptaxa[overlaptaxa!="sample-id"]
fgtkeep=pmp_fgt.columns[~pmp_fgt.columns.isin(overlapdrop)]
#subset out unique FGT columns
pmp_fgt=pmp_fgt[fgtkeep]
#combine unique FGT data with skin panel
pmp_combined=pmp_skin_data.merge(pmp_fgt, how="outer", on="sample-id")
#convert pmpmeta numeric data
#replace "Undefined" and nan values with 0
pmp_combined=pmp_combined.replace("UND", 0)
pmp_combined=pmp_combined.fillna(0)
pmp_combined=pmp_combined.apply(pd.to_numeric)
pmp_combined["sample-id"]=pmp_combined["sample-id"].astype('str')
pmpmeta=pmp_combined.merge(meta, how="inner", on="sample-id")

#drop extra buffers from weird day that you tried to process the ATCC mock community
pmpmeta=pmpmeta[pmpmeta["date_processed"]!="20240711"]

#split by method
pmp_ps=pmpmeta[pmpmeta["method"]=="powersoil"]
pmp_rl=pmpmeta[pmpmeta["method"]=="readylyse_dil"]
pmp_zy=pmpmeta[pmpmeta["method"]=="zymobiomics"]

#trying to use the same color palette as shotgun metagenomics (species_pal) although I think this is incomplete

# ### visualize unfiltered data ###
# #powersoil
# pmp_ps_unfilt=hf.viz_unfilt_qpcr(pmp_ps, species_pal)
# pmp_ps_unfilt.get_figure().savefig("/Users/liebermanlab/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/2024_03_sample_processing/filter_fig_df/unfilt_qpcr_ps.png", bbox_inches="tight")
# #zymo
# pmp_zy_unfilt=hf.viz_unfilt_qpcr(pmp_zy, species_pal)
# pmp_zy_unfilt.get_figure().savefig("/Users/liebermanlab/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/2024_03_sample_processing/filter_fig_df/unfilt_qpcr_zy.png", bbox_inches="tight")

# #readylyse
# pmp_rl_unfilt=hf.viz_unfilt_qpcr(pmp_rl, species_pal)
# pmp_rl_unfilt.get_figure().savefig("/Users/liebermanlab/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/2024_03_sample_processing/filter_fig_df/unfilt_qpcr_rl.png", bbox_inches="tight")

#%%  #### FILTER QPCR DATA #####

### Powersoil ###

###finding threshold from mock samples ###
#input for the set threshold function: 
#slice only mock samples with and without metadata
pmp_ps_mock=pmp_ps[pmp_ps["sampletype"]=="lm_mock"]
mock_df=pmp_ps_mock.iloc[:,2:-15].apply(pd.to_numeric)
#list of species in mock community
qpcrspecies=["Cutibacterium acnes", "Staphylococcus aureus", "Staphylococcus epidermidis", "Escherichia coli"]
#use set_threshold helper function to generate 1 plot and the threshold to filter data
qpcr_ps_plot,qpcr_ps_thresh=hf.set_threshold_qpcr(pmp_ps_mock,mock_df,qpcrspecies)
qpcr_ps_plot.get_figure().savefig(work_dir+"/supp_figures/FigS4_qpcr_powersoil_set_threshold.png",bbox_inches="tight")

#use perc_retain function to generate 1 plot
#input datasets for each source and the threshold to plot as vertical line
#subset data by source prior to using function because different datasets have different columns of metadata
legdata=pmp_ps[pmp_ps["source"]=="leg"].iloc[:,2:-15].apply(pd.to_numeric)
foredata=pmp_ps[pmp_ps["source"]=="forehead"].iloc[:,2:-15].apply(pd.to_numeric)
airdata=pmp_ps[pmp_ps["source"]=="air"].iloc[:,2:-15].apply(pd.to_numeric)
bufferdata=pmp_ps[pmp_ps["source"]=="buffer"].iloc[:,2:-15].apply(pd.to_numeric)
qpcr_ps_perc_retain=hf.perc_retain_qpcr(legdata, foredata,bufferdata, airdata, qpcr_ps_thresh)
qpcr_ps_perc_retain.get_figure().savefig(work_dir+"/supp_figures/FigS4_qpcr_powersoil_perc_retain.png", bbox_inches="tight")

#use threshold from above to filter data

pmp_ps.iloc[:,2:-15]=pmp_ps.iloc[:,2:-15].apply(pd.to_numeric)
#replace values below threshold with 0
filt1_pmp_ps=pmp_ps.iloc[:,2:-15]
filt1_pmp_ps[filt1_pmp_ps<qpcr_ps_thresh]=0

#### ZYMOBIOMICS ###

###finding threshold from mock samples ###
#input for the set threshold function: 
#slice only mock samples with and without metadata
pmp_zy_mock=pmp_zy[pmp_zy["sampletype"]=="lm_mock"]
mock_df=pmp_zy_mock.iloc[:,2:-15].apply(pd.to_numeric)
#list of species in mock community
qpcrspecies=["Cutibacterium acnes", "Staphylococcus aureus", "Staphylococcus epidermidis", "Escherichia coli"]
#use set_threshold helper function to generate 1 plot and the threshold to filter data
qpcr_zy_plot,qpcr_zy_thresh=hf.set_threshold_qpcr(pmp_zy_mock,mock_df,qpcrspecies)
qpcr_zy_plot.get_figure().savefig(work_dir+"/supp_figures/FigS4_qpcr_zymo_set_threshold.png",bbox_inches="tight")

#use perc_retain function to generate 1 plot
#input datasets for each source and the threshold to plot as vertical line
#subset data by source prior to using function because different datasets have different columns of metadata
legdata=pmp_zy[pmp_zy["source"]=="leg"].iloc[:,2:-15].apply(pd.to_numeric)
foredata=pmp_zy[pmp_zy["source"]=="forehead"].iloc[:,2:-15].apply(pd.to_numeric)
airdata=pmp_zy[pmp_zy["source"]=="air"].iloc[:,2:-15].apply(pd.to_numeric)
bufferdata=pmp_zy[pmp_zy["source"]=="buffer"].iloc[:,2:-15].apply(pd.to_numeric)
#qpcr_zy_perc_retain=hf.perc_retain_qpcr(legdata, foredata,bufferdata, airdata, qpcr_zy_thresh)
#qpcr_zy_perc_retain.get_figure().savefig("/Users/liebermanlab/Dropbox (MIT)/Lieberman Lab/Personal lab notebooks/Laura Markey/low-biomass-samples/2024_03_sample_processing/filter_fig_df/qpcr_zymo_perc_retain.png", bbox_inches="tight")

#for zymo need to do it separately in order to only plot forehead sample percent retained

og_fore_total=foredata.sum(axis=1)
og_air_total=airdata.sum(axis=1)
og_buffer_total=bufferdata.sum(axis=1)
fore_filt_perc_retain=[]
air_filt_perc_retain=[]
buffer_filt_perc_retain=[]
thresh=[]
for n in np.arange(0,5000,100):
    foredata[foredata<n]=0
    airdata[airdata<n]=0
    bufferdata[bufferdata<n]=0
    fore_filt_perc_retain.append(foredata.sum(axis=1).div(og_fore_total))
    air_filt_perc_retain.append(airdata.sum(axis=1).div(og_air_total))
    buffer_filt_perc_retain.append(bufferdata.sum(axis=1).div(og_buffer_total))
    thresh.append(n)
fore_retaindf=pd.concat(fore_filt_perc_retain, axis=1)
fore_retaindf.loc["thresh"]=thresh
air_retaindf=pd.concat(air_filt_perc_retain,axis=1)
air_retaindf.loc["thresh"]=thresh
buff_retaindf=pd.concat(buffer_filt_perc_retain,axis=1)
buff_retaindf.loc["thresh"]=thresh
plotforeretain=fore_retaindf.T
plotairretain=air_retaindf.T
plotbuffretain=buff_retaindf.T
meltfore=plotforeretain.melt(id_vars="thresh")
meltair=plotairretain.melt(id_vars="thresh")
meltbuff=plotbuffretain.melt(id_vars="thresh")
#plot threshold against percent sample retained colored leg/fore head, blanks separate
fig2,ax = plt.subplots()
plt.rcParams["figure.figsize"] = (5,3)
sns.lineplot(data=meltair, x="thresh", y="value", color="#dddddd")
sns.lineplot(data=meltbuff, x="thresh", y="value", color="darkgrey")
sns.lineplot(data=meltfore, x="thresh", y="value", color="#911c1c")
plt.xlim(-0.5,5015)
plt.vlines(400, 0,1, color="purple")
plt.ylim(0,1)
plt.xlabel("")
plt.ylabel("")
#plt.ylabel("Fraction original sample retained")
#plt.xlabel("Threshold ratio abundance")
fore_l = mpatches.Patch(color='#911c1c', label='forehead')
buff_l = mpatches.Patch(color='darkgrey', label='extraction')
air_l= mpatches.Patch(color="#dddddd", label="collection")
plt.legend(handles=[air_l, buff_l, fore_l], loc="lower right")
plt.savefig(work_dir+"/supp_figures/FigS4_qpcr_zymo_fore_blank_perc_retain.png",bbox_inches="tight")
plt.show()

#use threshold from above to filter data

pmp_zy.iloc[:,2:-15]=pmp_zy.iloc[:,2:-15].apply(pd.to_numeric)
#replace values below threshold with 0
filt1_pmp_zy=pmp_zy.iloc[:,2:-15]
filt1_pmp_zy[filt1_pmp_zy<qpcr_zy_thresh]=0

### Ready Lyse ####

###finding threshold from mock samples ###
#input for the set threshold function: 
#slice only mock samples with and without metadata
pmp_rl_mock=pmp_rl[pmp_rl["sampletype"]=="lm_mock"]
mock_df=pmp_rl_mock.iloc[:,2:-15].apply(pd.to_numeric)
#list of species in mock community
qpcrspecies=["Cutibacterium acnes", "Staphylococcus aureus", "Staphylococcus epidermidis", "Escherichia coli"]
#use set_threshold helper function to generate 1 plot and the threshold to filter data
qpcr_rl_plot,qpcr_rl_thresh=hf.set_threshold_qpcr(pmp_rl_mock,mock_df,qpcrspecies)
qpcr_rl_plot.get_figure().savefig(work_dir+"/supp_figures/FigS4_qpcr_rl_set_threshold.png",bbox_inches="tight")

#use perc_retain function to generate 1 plot
#input datasets for each source and the threshold to plot as vertical line
#subset data by source prior to using function because different datasets have different columns of metadata
legdata=pmp_rl[pmp_rl["source"]=="leg"].iloc[:,2:-15].apply(pd.to_numeric)
foredata=pmp_rl[pmp_rl["source"]=="forehead"].iloc[:,2:-15].apply(pd.to_numeric)
airdata=pmp_rl[pmp_rl["source"]=="air"].iloc[:,2:-15].apply(pd.to_numeric)
bufferdata=pmp_rl[pmp_rl["source"]=="buffer"].iloc[:,2:-15].apply(pd.to_numeric)
qpcr_rl_perc_retain=hf.perc_retain_qpcr(legdata, foredata,bufferdata, airdata, qpcr_rl_thresh)
qpcr_rl_perc_retain.get_figure().savefig(work_dir+"/supp_figures/FigS4_qpcr_readylyse_perc_retain.png", bbox_inches="tight")

#use threshold from above to filter data

pmp_rl.iloc[:,2:-15]=pmp_rl.iloc[:,2:-15].apply(pd.to_numeric)
#replace values below threshold with 0
filt1_pmp_rl=pmp_rl.iloc[:,2:-15]

filt1_pmp_rl[filt1_pmp_rl<qpcr_rl_thresh]=0

#### combine all qPCR extraction methods into a single dataset ###
#combined filtered absolute abundance data
combined_qpcr_filt=pd.concat([filt1_pmp_ps,filt1_pmp_zy, filt1_pmp_rl])
#combine list of sample ids in the same order
comb_id=pd.concat([pmp_ps["sample-id"], pmp_zy["sample-id"], pmp_rl["sample-id"]])

#filtered absolute abundance with sample id to export #
combined_qpcr_filt["sample-id"]=comb_id
combined_qpcr_filt.to_csv(work_dir+"/data/output_read_filter_datasets/filt_qpcr_absolute_abundace_all.csv", index=False)

#just the skin taxa to compare extraction methods fairly
skin_taxa_filt_qpcr=combined_qpcr_filt[skinonlytaxa]
skin_taxa_filt_qpcr["sample-id"]=combined_qpcr_filt["sample-id"]
skin_taxa_filt_qpcr.to_csv(work_dir+"/data/output_read_filter_datasets/filt_qpcr_absolute_abundace_skin_panel.csv", index=False)

#converting to relative abundance - all targets
#drop taxa that are not above 0 in any samples
taxsum=combined_qpcr_filt.iloc[:,:-1].sum()
dropzero=pd.Series(taxsum[taxsum==0].index)
combined_qpcr_filt_drop=combined_qpcr_filt.drop(columns=dropzero)
#renormalize
combined_qpcr_filt_drop["total"]=combined_qpcr_filt_drop.iloc[:,:-1].sum(axis=1)
combined_qpcr_filt_renorm=combined_qpcr_filt_drop.iloc[:,:-2].div(combined_qpcr_filt_drop["total"], axis=0)
#add back metadata
combined_qpcr_filt_renorm["sample-id"]=comb_id
combined_qpcr_filt_renorm.to_csv(work_dir+"/data/output_read_filter_datasets/filt_qpcr_rel_abundance_all.csv", index=False)

#converting to relative abundance - skin taxa only
#drop taxa that are not above 0 in any samples
taxsum=skin_taxa_filt_qpcr.iloc[:,:-1].sum()
dropzero=pd.Series(taxsum[taxsum==0].index)
skin_taxa_filt_qpcr_drop=skin_taxa_filt_qpcr.drop(columns=dropzero)
#renormalize
skin_taxa_filt_qpcr_drop["total"]=skin_taxa_filt_qpcr_drop.iloc[:,:-1].sum(axis=1)
skin_taxa_filt_qpcr_renorm=skin_taxa_filt_qpcr_drop.iloc[:,:-2].div(skin_taxa_filt_qpcr_drop["total"], axis=0)
#fill nan values with zeros
skin_taxa_filt_qpcr_renorm=skin_taxa_filt_qpcr_renorm.fillna(0)
#add back metadata
#drop from analysis rows with 0 signal
skin_taxa_filt_qpcr_renorm["sample-id"]=comb_id
skin_taxa_filt_qpcr_renorm=skin_taxa_filt_qpcr_renorm[skin_taxa_filt_qpcr_renorm.iloc[:,:-1].sum(axis=1)>0]
skin_taxa_filt_qpcr_renorm.to_csv(work_dir+"/data/output_read_filter_datasets/filt_qpcr_rel_abundance_skin_panel.csv", index=False)

