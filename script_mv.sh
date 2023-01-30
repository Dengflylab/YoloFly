rm png2/classes.txt

for i in $(ls png2/*| sed 's/.txt//;s/.png//'|sort|uniq|head -n 40);do mv $i.png fly/train/images; mv $i.txt fly/train/labels; done

for i in $(ls png2/*| sed 's/.txt//;s/.png//'|sort|uniq|head -n 5);do mv $i.png fly/test/images; mv $i.txt fly/test/labels; done
for i in $(ls png2/*| sed 's/.txt//;s/.png//'|sort|uniq|head -n 10);do mv $i.png fly/valid/images; mv $i.txt fly/valid/labels; done


# training 70% 40
# test 10%     5
# valide 20%   10
