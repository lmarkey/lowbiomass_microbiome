#QIIME2 workflow

#activate local conda installation of qiime
conda activate qiime2

#import sequences from sample manifest file
qiime tools import \
--type 'SampleData[SequencesWithQuality]' \
--input-path local_manifest.txt \
--output-path single-end-f-demux.qza \
--input-format SingleEndFastqManifestPhred33V2 


#checking quality and number of reads

#need to trim 16s primers
qiime cutadapt trim-single \
  --i-demultiplexed-sequences single-end-f-demux.qza \
  --p-front AGAGTTTGATCMTGGCTCAG \
  --o-trimmed-sequences single-end-f-demux-trimmed.qza

#check on read quality and compare trimmed and untrimmed

qiime demux summarize \
--i-data single-end-f-demux-trimmed.qza \
--o-visualization single-end-f-trimmed.qzv


qiime demux summarize \
--i-data single-end-r-demux.qza \
--o-visualization single-end-r.qzv


#quality looks pretty good out to 130
#cluster analysis using previous pipeline - dada2 and classifier
qiime dada2 denoise-single \
  --i-demultiplexed-seqs single-end-f-demux-trimmed.qza  \
  --p-trunc-len 130 \
  --output-dir dada2output_trimmed

#classify using classifier to get species-level resoluation of Cutibacterium 
#made a new classifier using rescript, qual filter and dereplicating
qiime feature-classifier classify-sklearn \
  --i-classifier rescript_class/silva-138.1-ssu-nr99-27f-534r-classifier.qza \
  --i-reads dada2output_trimmed/representative_sequences.qza \
  --output-dir dada2_trimmed_classifier_output 



