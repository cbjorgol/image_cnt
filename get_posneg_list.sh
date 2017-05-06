 find ../train_byclass/None/ -iname "*.png" > neg.txt
 find ../train_byclass/adult_females/ -iname "*.png" > pos.txt
 find ../train_byclass/adult_males/ -iname "*.png" >> pos.txt
 find ../train_byclass/juveniles/ -iname "*.png" >> pos.txt
 find ../train_byclass/pups/ -iname "*.png" >> pos.txt
 find ../train_byclass/subadult_males/ -iname "*.png" >> pos.txt
