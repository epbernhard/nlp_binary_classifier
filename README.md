**Binary classification using NLP**

This presents a binary classifier that uses NLP to classify comments reviewing healthcare practices. I stress that it has been tailored for a given dataset, and is only presented here as part of my portfolio. I will update it regularly to test new models, every time trying to push its performances.

---
On this page, you can find all the codes, which should only be useful to me to keep track of the changes, but feel free to have a look if you are nosy. Instead, the latest version of the classifier is described in details in a set of notebooks that you can find in the "notebooks" folder. I divided the task into three sub-tasks, which can be individually improved to increase the performances of the classifier. These are,

```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```

<span style="color:lightgreen">**Data Preparation (prep_data.ipynb)**:</span><br>
In this notebook you can find a detailed description on how I explore the data (e.g. imbalance), as well as how I define a vocabulary.

<span style="color:lightgreen">**Word Embedding (word_embedding.ipynb)**:</span><br>
In this notebook you can find a detailed description on which embedding I use to convert the comments from plain English to vectors ready to be ingested by the NLP algorithm.

<span style="color:lightgreen">**Model (nlp_model.ipynb)**:</span><br>
In this notebook you can find a detailed description of the model I use to fit the data, and a little summary of its performances.

I will work on each of these, and if there is some progress on the classification I will update the notebooks to reflect the changes I have made.

---
<span style="color:red">**Cautionary Notes**</span><br>
<ol>
  <li>The data will note be shared, as coming from private communication and sensitive.</li>
  <li>The classifier is not intended for production, this is only part of my portfolio. However, feel free to use any parts of it if you need.</li>
  <li>The versioning does not follow the PEP 440, it is only used as a self-guide to keep track of the changes.</li>
</ol>

