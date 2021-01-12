import os
import sys
import subprocess
from os import path
import pandas as pd
import numpy as np
import math

input_file_1 = sys.argv[1]
input_file_2 = sys.argv[2]
#input_file_3 = sys.argv[3]



file_path = os.path.split(input_file_2)
output_file_1 = file_path[0] + "/" + "rsID_pull_phecode-696.41-PsC_both_sexes_with_ORs_sorted_with_Variant_Information_genotype_count.csv"
#output_file_2 = file_path[0] + "/" + "phecode-696.42-PsA_both_sexes_with_P-value_sorted.csv"
#output_file_2 = file_path[0] + "/" + "L12_PSORI_ARTHRO_vs_L12_PSORIASIS_with_Variant_information_OR_Sorted_Descending_Top_1000.csv"
# output_file_1 = file_path[0] + "/" + "Chr_1-6_L12_PSORI_ARTHRO_vs_L12_PSORIASIS_OR_2.5_grt_AF_grt_0.01_variants.csv"
# output_file_2 = file_path[0] + "/" + "Chr_7-14_L12_PSORI_ARTHRO_vs_L12_PSORIASIS_OR_2.5_grt_AF_grt_0.01_true_confident_variants.csv"
# output_file_3 = file_path[0] + "/" + "Chr_15-22_L12_PSORI_ARTHRO_vs_L12_PSORIASIS_OR_2.5_grt_AF_grt_0.01_true_confident_variants.csv"
# output_file_4 = file_path[0] + "/" + "Chr_X_L12_PSORI_ARTHRO_vs_L12_PSORIASIS_OR_2.5_grt_AF_grt_0.01_true_confident_variants.csv"

# output_file_1 = file_path[0] + "/" + "Chr_14-16_UKBB_icd10_M07_PsA_Case_AF_grt_0.01_variants.csv"
# output_file_2 = file_path[0] + "/" + "Chr_4-6_UKBB_icd10_M07_PsA_Case_AF_grt_0.01_variants.csv"
# output_file_3 = file_path[0] + "/" + "Chr_7-10_UKBB_icd10_M07_PsA_Case_AF_grt_0.01_variants.csv"
# output_file_4 = file_path[0] + "/" + "Chr_X_UKBB_icd10_M07_PsA_Case_AF_grt_0.01_variants.csv"
# output_file_5 = file_path[0] + "/" + "Chr_11-13_UKBB_icd10_M07_PsA_Case_AF_grt_0.01_variants.csv"
# output_file_6 = file_path[0] + "/" + "Chr_14-16_UKBB_icd10_M07_PsA_Case_AF_grt_0.01_variants.csv"
# output_file_7 = file_path[0] + "/" + "Chr_17-19_UKBB_icd10_M07_PsA_Case_AF_grt_0.01_variants.csv"
# output_file_8 = file_path[0] + "/" + "Chr_20-22_UKBB_icd10_M07_PsA_Case_AF_grt_0.01_variants.csv"


#output_file_1 = file_path[0] + "/" + "EUR_PsC_ variants_absent_in_PsAwithPsC_with_1000G_Annotations_BioMe.csv"
# output_file_2 = file_path[0] + "/" + "PsA_Common_Variants_with_PsO.xlsx"
# output_file_3 = file_path[0] + "/" + "PsO_Common_Variants_with_PsO_and_PsOandPsA_BioME_cases.xlsx"


# The below line will read the provided input file in excel format
#df_1 = pd.read_excel(input_file_1)
# df_2 = pd.read_excel(input_file_2, sep=',', low_memory=False)
df_1 = pd.read_csv(input_file_1, sep='\t',low_memory=False)
df_2 = pd.read_csv(input_file_2, sep=',', low_memory=False)

# df_2 = pd.read_csv(input_file_2, sep=',', low_memory=False)
#df_2.columns = ['CHR', 'POS', 'SNP', 'RISK', 'NR','case','control','all','OR','ln(OR)', 'LB', 'UB', 'Zscore','p-value', 'direction']
#print(df_2)
#df_3 = pd.read_excel(input_file_3)
# The below line of code will only pull out the matching variants between two dataframe
# df_1_subset = df_1[df_1.rsid.isin(df_2.rsid)]
# df_2_subset = df_2[df_2.POS.isin(df_1.POS)]
# The below line of code will pull out non matching variants from df1
# df_1_subset = df_1[~df_1.POS.isin(df_2.POS)]
#
# df_1_subset.to_csv(output_file_1, index= 0)

# The below line of code will pull out the matching variants from df1, df2 and df3
#df_2_subset = df_1[df_1.POS.isin(df_2.POS)  & df_1.POS.isin(df_3.POS)]
#print(df_2_subset)

# df_2_subset.to_csv(output_file_1)



# output_file_base = os.path.splitext(output_text_file)[0]
# hwe_test_file = output_file_base + "_HWE"
# print(hwe_test_file)
# logfile = output_file_base + "_logfile.txt"
# print(logfile)
# hwe_input_file = input_vcf_file


# The below lines of code can be used to remove the rows with NA values and to sort the files based on values in specified column
# df = pd.read_csv(input_file_1, sep='\t', low_memory=False)
# df[['pval_EUR']] = df[['pval_EUR']].apply(pd.to_numeric, errors='coerce')
# # df = df.dropna()
# final_df = df.sort_values('pval_EUR')
# final_df.to_csv(output_file_1)

# df_1 = pd.read_csv(input_file_1, sep=',', low_memory=False)
# df_2 = pd.read_csv(input_file_2, sep='\t', low_memory=False)
# df_1_subset = df_1[df_1.variant.isin(df_2.variant)]
# df_1_subset.to_csv(output_file_1, index=0)
# This line of the code can be used to merge the PsA and PsC cases from GWAS data from UKBB
# df_2.drop(df_2.columns[[6,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]], axis=1, inplace=True)
#merge_file = (pd.merge(df_1, df_2, on=['chr','pos','ref','alt'], how='outer'))

#merge_file = (pd.merge(df_1, df_2, on=['chr','pos','ref','alt'], how='right'))


#merge_file = (pd.merge(df_1, df_2, on=['CHROM','POS','REF','ALT'], how='inner'))
# merge_file = (pd.merge(df_1, df_2, on=['avsnp150'], how='inner'))
# merge_file.to_csv(output_file_1, index=0)


# merge_file = (pd.merge(df_1, df_2, on=['variant'], how='outer'))
# merge_file.drop(merge_file.columns[[6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], axis=1, inplace=True)
# merge_file = (pd.merge(df_1, df_2, on=['rsid'], how='left'))
# merge_file = (pd.merge(df_1, df_2, on=['rsid'], how='inner'))
merge_file = (pd.merge(df_1, df_2, on=['rsid'], how='right'))
merge_file.to_csv(output_file_1, index= 0)
# merge_file['OR_PsAvsPsC'] = merge_file ['PsA_af_cases_EUR']/ merge_file['PsC_af_cases_EUR']
# merge_file['OR_PsAvsPsC'] = merge_file['OR_PsAvsPsC'].astype(float)
#This line of the code will remove the line with NaN and inf values from the pandas dataframe
# merge_file_1 = merge_file[~merge_file.isin([np.nan, np.inf, -np.inf]).any(1)]
# merge_file_2 = merge_file_1.loc[(merge_file_1['OR_PsAvsPsC'] >= 2.5)]

# This line of the code can be used to merge the PsA and PsC cases from GWAS data from NealLab
#merge_file = (pd.merge(df_1, df_2, on=['variant','minor_allele'], how='left'))
# merge_file = (pd.merge(df_1, df_2, on=['variant'], how='left'))
# df_1_1 = df_1.sort_values(by='OR_PsAvsPsC')
# df_1_1.to_csv(output_file_1, index=0)
# df_1['PsC_Odds_Ratio'] = np.exp(df_1['PsC_beta_EUR'])
# merge_file = (pd.merge(df_1, df_2, on=['chr','pos','ref','alt'], how='left'))
# merge_file.to_csv(output_file_1, index=0)
# df_1_1 = df_1.sort_values(by='PsA_Odds_Ratio', ascending=False)
# df_1_2 = df_1.sort_values(by='PsA_pval_EUR', ascending=True)
# df_1_1.to_csv(output_file_1, index=0)
# df_1_2.to_csv(output_file_2, index=0)
#print(df_1_1)
# merge_file['PsA_Odds_ratio'] = np.exp(merge_file['PsA_beta'])
# merge_file['PsC_Odds_ratio'] = np.exp(merge_file['PsC_beta'])
# merge_file['OR_PsAvsPsC'] = merge_file['PsA_Odds_ratio'] / merge_file['PsC_Odds_ratio']
# merge_file['OR_PsAvsPsC'] = merge_file['OR_PsAvsPsC'].astype(float)
#merge_file['OR_PsAvsPsC'] = pd.numeric(merge_file.OR_PsAvsPsC, errors = 'coerce')
#This line of the code will remove the line with NaN and inf values from the pandas dataframe
# merge_file_1 = merge_file[~merge_file.isin([np.nan, np.inf, -np.inf]).any(1)]
# merge_file_2 = merge_file_1.loc[(merge_file_1['OR_PsAvsPsC'] >= 2.5)]
# merge_file.to_csv(output_file_1, index=0)

# merge_file_2.to_csv(output_file_2, index=0)
#merge_file_2 = merge_file[merge_file['OR_PsAvsPsC'>= 2.5]]
#merge_file_2.to_csv(output_file_1, index=0)

# The below lines of code will read the CSV file and filter it with given criteria
# df_1 = pd.read_csv(input_file_1, sep=',', low_memory=False)
# df_1['PsA_minor_AF'] = pd.to_numeric(df_1.PsA_minor_AF, errors='coerce')
# It selects the variants with case_AF greater than equal to 0.01 (1%)
# df_1_1 = df_1.loc[(df_1['PsA_minor_AF'] >= 0.01)]
# df_1_1.drop(df_1_1.columns[[6,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]], axis=1, inplace=True)
# df_1_1.to_csv(output_file_1, index= 0)
# df_X = df_1_1[(df_1_1['chr'] == 'X')]
# df_X.to_csv(output_file_4)
# df_1_1['chr'] = pd.to_numeric(df_1_1.chr, errors='coerce')
# df_chr_1 = df_1_1.loc[(df_1_1['chr'] <= 1)]
# df_chr_9 = df_1_1.loc[(df_1_1['chr'] == 9)]
# df_chr_14_16 = df_1_1.loc[(df_1_1['chr'] <= 16) & (df_1_1['chr'] >= 14)]
# df_4_5 = df_1_1.loc[(df_1_1['chr'] <= 5) & (df_1_1['chr'] >= 4)]
# df_7_10 = df_1_1.loc[(df_1_1['chr'] <= 10) & (df_1_1['chr'] >= 7)]
# df_11_13 = df_1_1.loc[(df_1_1['chr'] <= 13) & (df_1_1['chr'] >= 11)]
# df_14_16 = df_1_1.loc[(df_1_1['chr'] <= 16) & (df_1_1['chr'] >= 14)]
# df_17_19 = df_1_1.loc[(df_1_1['chr'] <= 19) & (df_1_1['chr'] >= 17)]
# df_20_22 = df_1_1.loc[(df_1_1['chr'] <= 22) & (df_1_1['chr'] >= 20)]
# df_chr_14_16.to_csv(output_file_1)
# df_4_6.to_csv(output_file_2)
# df_7_10.to_csv(output_file_3)
# df_11_13.to_csv(output_file_5)
# df_14_16.to_csv(output_file_6)
# df_17_19.to_csv(output_file_7)
# df_20_22.to_csv(output_file_8)

#The below lines of code will read the CSV file and filter it with given criteria
# df_1 = pd.read_csv(input_file_1, sep=',', low_memory=False)
# df_1['PsA_af_controls_EUR'] = pd.to_numeric(df_1['PsA_af_controls_EUR'])
# df_1_1 = df_1.loc[(df_1['PsA_af_controls_EUR'] >= 0.005) & (df_1['PsA_low_confidence_EUR'] == 'True') ]
#It selects the variants with control_AF greater than equals to 0.005 (0.5%)
# df_1_1 = df_1.loc[(df_1['PsA_af_controls_EUR'] >= 0.005)]
# df_1_2 = df_1_1.loc[(df_1['PsA_low_confidence_EUR'] == 'True')]
#Print the variants from chromosome X before converting that column to numeric
# df_X = df_1_1[(df_1_1['chr'] == 'X')]
# df_X.to_csv(output_file_4)
# df_1_1['chr'] = pd.to_numeric(df_1_1.chr, errors='coerce')
# df_1_6 = df_1_1.loc[(df_1_1['chr'] <= 6)]
# df_7_14 = df_1_1.loc[(df_1_1['chr'] <= 14) & (df_1_1['chr'] >= 7)]
# df_15_22 = df_1_1.loc[(df_1_1['chr'] <= 22) & (df_1_1['chr'] >= 15)]

# df_1_6.to_csv(output_file_1)
# df_7_14.to_csv(output_file_2)
# df_15_22.to_csv(output_file_3)

#The below lines of code will read the CSV file and filter it with given criteria
# df_1 = pd.read_csv(input_file_1, sep=',', low_memory=False)
# merge_file_2['PsA_af_controls_EUR'] = pd.to_numeric(merge_file_2['PsA_af_controls_EUR'])
#df_1_1 = merge_file_2.loc[(df_1['PsA_af_controls_EUR'] >= 0.01) & (merge_file_2['PsA_low_confidence_EUR'] == 'True') ]
#It selects the variants with control_AF greater than equals to 0.01 (1%)
# df_1_1 = merge_file_2.loc[(merge_file_2['PsA_af_controls_EUR'] >= 0.01)]
# df_1_2 = df_1_1.loc[(df_1['PsA_low_confidence_EUR'] == 'True')]
#Print the variants from chromosome X before converting that column to numeric
# df_X = df_1_1[(df_1_1['chr'] == 'X')]
# df_X.to_csv(output_file_4)
# df_1_1['chr'] = pd.to_numeric(df_1_1.chr, errors='coerce')
# df_1_6 = df_1_1.loc[(df_1_1['chr'] <= 6)]
# df_7_14 = df_1_1.loc[(df_1_1['chr'] <= 14) & (df_1_1['chr'] >= 7)]
# df_15_22 = df_1_1.loc[(df_1_1['chr'] <= 22) & (df_1_1['chr'] >= 15)]
# df_1_6.to_csv(output_file_1)
# df_7_14.to_csv(output_file_2)
# df_15_22.to_csv(output_file_3)


#The below lines of code will read the CSV file and filter it with given criteria
# df_1 = pd.read_csv(input_file_1, sep=',', low_memory=False)
# merge_file_2['PsA_minor_AF'] = pd.to_numeric(merge_file_2['PsA_minor_AF'])
#df_1_1 = merge_file_2.loc[(df_1['PsA_af_controls_EUR'] >= 0.01) & (merge_file_2['PsA_low_confidence_EUR'] == 'True') ]
#It selects the variants with control_AF greater than equals to 0.01 (1%)
# df_1_1 = merge_file_2.loc[(merge_file_2['PsA_minor_AF'] >= 0.01)]
# df_1_2 = df_1_1.loc[(df_1['PsA_low_confidence_EUR'] == 'True')]
#Print the variants from chromosome X before converting that column to numeric

# df_X = df_1_1[(df_1_1['chr'] == 'X')]
# df_X.to_csv(output_file_4)
# df_1_1['chr'] = pd.to_numeric(df_1_1.chr, errors='coerce')
# df_1_6 = df_1_1.loc[(df_1_1['chr'] <= 6)]
# df_7_14 = df_1_1.loc[(df_1_1['chr'] <= 14) & (df_1_1['chr'] >= 7)]
# df_15_22 = df_1_1.loc[(df_1_1['chr'] <= 22) & (df_1_1['chr'] >= 15)]
# df_1_6.to_csv(output_file_1)
# df_7_14.to_csv(output_file_2)
# df_15_22.to_csv(output_file_3)




