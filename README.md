# SCZ-EF-stacked-classifiers

This repository contains the source code accompanying our paper "Leveraging Stacked Classifiers for Multi-task Executive Function in Schizophrenia Yields Diagnostic and Prognostic Insights," published in *Schizophrenia Bulletin*.

![Study Overview](Figure%201.%20Study%20overview.png)
*Figure 1. Overview of the study design and analytical framework.*

---

## Abstract

**Background:** Executive functioning (EF) impairments are often seen in mental disorders, particularly schizophrenia, where they relate to adverse outcomes. As a heterogeneous construct, how specifically each dimension of EF characterizes the diagnostic and prognostic aspects of schizophrenia remains opaque.

**Study Design:** We used classification models with a stacking approach on systematically measured EFs using 6 tasks to discriminate 195 patients with schizophrenia from healthy individuals. Baseline EF measurements were moreover employed to predict symptomatically remitted or non-remitted prognostic subgroups. EF feature importance was determined at the group level, and the ensuing individual importance scores were associated with four symptom dimensions.

**Study Results:** The models highlighted the importance of inhibitory control (interference and response inhibitions) and working memory in accurately identifying individuals with schizophrenia (area under the curve [AUC]=0.87) and those in remission (AUC=0.81). Patients who were correctly classified, in association with the contribution of interference inhibition function to our diagnostic classifier, presented more severe baseline negative symptoms compared to those who were more likely to be misclassified. Also, linked to the function of working memory updating, patients who were successfully classified as remitted displayed milder cognitive symptoms at follow-up. Remitted patients did not differ significantly from non-remitted cases in baseline EF assessments or overall symptom severity.

**Conclusions:** Our work indicates that impairments in specific EF dimensions in schizophrenia are differentially linked to individual symptom load and prognostic outcomes. Thus, assessments and models based on EF may be promising in the clinical evaluation of this disorder.

**Keywords:** Schizophrenia; Executive functions; Machine learning; Prognostic prediction

---

## Main Findings

### Executive Function Assessments
![Multi-task Executive Function](figures/Figure%202.%20Multi-task%20executive%20function%20dimension%20assessments.png)
*Figure 2. Multi-task executive function dimension assessments.*

### Diagnostic Models
![Diagnostic Models](figures/Figure%203.%20Performance%20and%20feature%20importance%20for%20diagnostic%20models.png)
*Figure 3. Performance and feature importance for diagnostic models.*

### Prognostic Models
![Prognostic Models](figures/Figure%204.%20Performance%20and%20feature%20importance%20for%20prognostic%20models.png)
*Figure 4. Performance and feature importance for prognostic models.*

### Feature-Symptom Correlations
![Feature-Symptom Correlations](figures/Figure%205.%20Correlation%20between%20the%20importance%20of%20an%20executive%20function%20feature%20and%20individual%20psychopathology%20along%20four%20symptom%20dimensions.png)
*Figure 5. Correlation between the importance of an executive function feature and individual psychopathology along four symptom dimensions.*

---

## Citation

Zhang, T., Zhao, X., Yeo, B. T. T., Huo, X., Eickhoff, S. B., & Chen, J. (2024). Leveraging Stacked Classifiers for Multi-task Executive Function in Schizophrenia Yields Diagnostic and Prognostic Insights. *medRxiv*. https://doi.org/10.1101/2024.12.05.24318587

**Note:** This preprint is currently under review at *Schizophrenia Bulletin*. Citation information will be updated upon publication.
