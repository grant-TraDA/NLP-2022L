# Model (accent prediction) after 100 epochs:

Input first line: ```Once upon a midnight dreary```

Input part of second line: ```while I pondered```

Accents detected in first line
```a aA a Aa A A Aa no_accent no_accent ...```

Accents detected in second line ```a A A  no_accent no_accent ...```

```
Best candidates for next word:
        A: 99.43318367004395%
        a: 0.5643955431878567%
        Aa: 0.0022517751858686097%
        : 0.00016407100247306516%
        aA: 6.774691918565168e-10%
```
---

Input first line: ```Stasis in darkness. Then the substanceless blue```

Input part of second line: ```Pour of tor and distances.```

Accents detected in first line
```A  a Aa  A a Aa A  no_accent no_accent ...```

Accents detected in second line
```A a a A Aaa  no_accent no_accent ...```

```
Best candidates for next word:
        : 83.28053951263428%
        A: 16.673557460308075%
        a: 0.04590644966810942%
        Aa: 9.097684383618798e-07%
        aA: 1.4540406097045266e-11%
```
---

Input first line: ```Do not go gentle into that good night,```

Input part of second line: ```Old age should burn```

Accents detected in first line
```A A A Aa aa A a A  no_accent no_accent ...```

Accents detected in second line
```A A A  no_accent no_accent no_accent ...```

```
Best candidates for next word:
        A: 99.9380350112915%
        a: 0.06134043214842677%
        : 0.0005922586296946974%
        Aa: 2.255896447422856e-05%
        aA: 1.9488731375222335e-10%
```
Lines are padded to 20 tokens, so ```...``` means a lot of ```no_accent``` because of the padding.

# Model (accent prediction) after 1 epoch:
Here, the network just approximated the accent distribution in the dataset
![obraz](https://user-images.githubusercontent.com/47048420/168491842-95fe0e91-bee5-4b4d-806c-4aae67535b55.png)


Examples are the same
```
Best candidates for next word:
        A: 99.99994039535522%
        : 4.6810811227260274e-05%
        a: 6.354833459454312e-06%
        Aa: 7.753644837293905e-07%
        aA: 2.9514532704226666e-13%
```
---
```
Best candidates for next word:
        A: 99.99063014984131%
        : 0.0093532566097565%
        a: 1.2933040238749527e-05%
        Aa: 2.850233070395003e-10%
        aA: 1.5458732824448521e-16%
```
---
```
Best candidates for next word:
        A: 99.99063014984131%
        : 0.0093532566097565%
        a: 1.2933040238749527e-05%
        Aa: 2.850233070395003e-10%
        aA: 1.5458732824448521e-16%
```

# Gpt2 fine-tuning:
## before:

Input first line: ```Once upon a midnight dreary```

Input part of second line: ```while I pondered```

High temperature: ```while I pondered over the details of my past, my heart began to throb and I felt my life getting closer```

Low temperature: ```while I pondered the fate of the world, I was struck by the sudden and unexpected change in the world.```

---

Input first line: ```Stasis in darkness. Then the substanceless blue```

Input part of second line: ```Pour of tor and distances.```

High temperature: 
```
Pour of tor and distances.


"Let us go. This is the place. The whole place. It is
```
Low temperature: ```Pour of tor and distances. Then the substanceless blue Pour of tor and distances. Then the substanceless blue Pour of tor```

---

Input first line: ```Do not go gentle into that good night,```

Input part of second line: ```Old age should burn```

High temperature:
```
Old age should burn you."


The sun is not set. It is only at sunrise, when the
```

Low temperature:
```
Old age should burn you to the ground.


"I will not go to sleep until you have been
```


## after fine-tuning:
Input first line: ```Once upon a midnight dreary```

Input part of second line: ```while I pondered```

High temperature: ```while I pondered on a golden news report on scary sea whose flesh and on your hand for grief whom out laughter```

Low temperature: ```while I pondered gives again all to me and any rather burn fire generous nest of gold and praise thee and on```

---

Input first line: ```Stasis in darkness. Then the substanceless blue```

Input part of second line: ```Pour of tor and distances.```

High temperature: ```Pour of tor and distances. by air through thick count brit field though their soft and by way so long enough for a```

Low temperature: ```Pour of tor and distances. by a single count or a single finger crime or heaven you may and by your ways to me```

---

Input first line: ```Do not go gentle into that good night,```

Input part of second line:  ```Old age should burn```

High temperature: ```Old age should burn up todoes something extraordinary all that touches him with their own thought room is most ones heart shaped```

Low temperature: ```Old age should burn up that first time for you to try to eat them very smile and dear lord and do not```

Fine-tuning entirely wiped out newlines and generally changed the vocabulary and grammatical structure.


# (Model+fine-tuned gpt2) vs (fine-tuned gpt2)

First line: ```when you come round a corner and```

High temperature:

  With model: ```ive no time to stop it my friends will id be as bold as you can be as you```

  No model:    ```to be acknowledged never a one has a word is spoken to you she is no more brave for```
        
Low temperature:

  With model: ```ive no doubt id out just be dead for all my fun shell be not my hero i is```

  No model:    ```for to have a ball between them they have no choice but to have him in them it a```

---

First line: ```centuries ago some built fires in caves of stone```

High temperature:

  With model: ```ive found a world of dread ive by ur side a british or rome or```

  No model:    ```bloom in grown fat gown and multi games with various teaching skill in their kids gown and stamps out```
        
Low temperature:

  With model: ```ive found a free form of space for us in an infinite unique rome our an out right```

  No model:    ```bloom in rage of grief and sorrow sin i cried for pain far calm memory for grief and grief```

---

First line: ```cool to look to something outside```

High temperature:

  With model: ```ive meet my odd odd odd hair may be found atans part in my heart so now i```

  No model:    ```yourself something else very wrong something greater something else will happen that i seem bad to be in hot```
        
Low temperature:

  With model: ```ive meetin you in heaven all i know is i in heaven all my bread is so deep```

  No model:    ```yourself something else to abuse something to set you straight down that deeds and you will not see again```

---

First line: ```as large as a tank aiming```

High temperature:

  With model: ```ive notfound in that world of art work in sight of their own world for us we care```

  No model:    ```whether to leave whether truth or substance failed so please give us bear it forever perhaps some say it```
        
Low temperature:

  With model: ```ive run out of time i am young as age men are of old we know now be a```

  No model:    ```at ride its not a story that we all paint of him as a memory for him to be```


Overall, the model seems to boost one-syllable words. The tendency to largely boost probability of "I've" at the very beginning has not been explained yet. 
