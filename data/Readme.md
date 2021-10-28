# Data Format
## Sentence Classification
---
 ### Format
          <sent>+"\t\t"+<label>

### Statistics


  | Dataset        | Total | Positive | Negative | LEN |
  |-----------------|:---------:|:---------------:|:--------------:|:--------:|
  | train_auto | 421,776 | 210,888  |  210,888  |29.8 
  | train_manual     |   5028    |    2117    |     2911  | 29.2
  | dev     |    1066   |    431    |     635    | 30.1
  | test      |    2104    |    852    |    1252   | 30.1

  
## Tagging

----
### Format (five rows per instance)
     <sent> + " ||| " + <classify_label> 
     <Synthesized POS> + " ||| " + <"Original"/"Neutral"> 
     <Synthesized POS> + " ||| " + <"Original"/"Neutral"> 
     <token_tags> + " ||| " + <"Original"/"Neutral"> 
     "\n"
     
- <b>sent</b> are tokens connected with space. 

- <b>token_tags</b> include: "B-bias", "I-bias", "O" and follows the BIO format.

- <b><"Original"/"Neutral"></b> implies whether this sent is from the source side or target side of a revision pair.
    

### Automatic data
- <b>noisy_train*.tsv</b> 

    source side of Wikipedia revision sentence pairs with automatically generated token labels

- <b>noisy_train_neg.tsv</b> 

    target side of Wikipedia revision sentence pairs with automatically generated token labels