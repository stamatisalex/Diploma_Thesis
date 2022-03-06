#!/bin/bash
#gt
find data/acdc/gt/*/train/* -name "*gt_labelIds.png" |sort -n| cut -d '/' -f3- > data/list/acdc/seperate_files/gt_train.lst
find data/acdc/gt/*/val/* -name "*gt_labelIds.png" |sort -n| cut -d '/' -f3- > data/list/acdc/seperate_files/gt_val.lst
#rgb_anon
find data/acdc/rgb_anon/*/test/* -name "*.png" |sort -n| cut -d '/' -f3- > data/list/acdc/test.lst
find data/acdc/rgb_anon/*/test_ref/* -name "*.png" |sort -n| cut -d '/' -f3- > data/list/acdc/test_ref.lst
find data/acdc/rgb_anon/*/train/* -name "*.png" |sort -n| cut -d '/' -f3- > data/list/acdc/seperate_files/rgb_anon_train.lst
find data/acdc/rgb_anon/*/train_ref/* -name "*.png" |sort -n| cut -d '/' -f3- > data/list/acdc/seperate_files/rgb_anon_train_ref.lst
find data/acdc/rgb_anon/*/val/* -name "*.png" |sort -n| cut -d '/' -f3- > data/list/acdc/seperate_files/rgb_anon_val.lst
find data/acdc/rgb_anon/*/val_ref/* -name "*.png" |sort -n| cut -d '/' -f3- > data/list/acdc/seperate_files/rgb_anon_val_ref.lst

