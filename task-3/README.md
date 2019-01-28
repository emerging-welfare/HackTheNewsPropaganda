We have used a keyword counter based system to detect keywords that occur frequently in the training dataset for each label.

For each label frequently occurring words are extracted from the training set.
Secondly, these extracted keywords are given weights according to their orthographic features (capitalization etc.) and frequency of occurrence inside the dataset. In order to obtain stable results the log value of the inverse of the frequency is used for each keyword.
During prediction we have used a sentence level approach and evaluated the score of each label on a single sentence. This approach has the advantage of obtaining fast results with a tradeoff for ignoring the predictions made for the neighboring sentence. Highest scoring label is returned as the prediction for a given fragment.


In order to replicate our results, you can use the ipython notebook included in this folder.
Dependencies: nltk library

Please run this additional code if receive errors due to nltk.stopwords

```
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

```

