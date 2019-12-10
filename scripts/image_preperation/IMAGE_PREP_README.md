# Image Preperation
## Idea
To get a good training dataset that applies in real situations we want to exchange the white background of basic cable images and replace them with random backgrounds.
In addition to that mirrored images and images from differen perspectives of cables should give a solid ground for the CV system to recognize the type of cable.

## Useage
- Step1: Copy images (max 35 in one bach) into '/images_to_change'
- Step2: execute 'py prepare_images' on your terminal in the image_preperation folder
- Step3: copy generated images from one of the '/converted_images_{X}' directories
- Step4: delete the generated '/converted_images_{X}' folder

## Notes
When pushing to Github
- please do not leave any images in the skript folder. With exeption of the images in 'scripts/image_preperation/random_backgrounds/' folder.
- please do not leave generated directories in the skript folders
we do not want to litter all over the reposetory