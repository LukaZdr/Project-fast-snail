# -*- coding: utf-8 -*-

csv = open("Results.csv","r")
html = open("Results.html", "w")

html.write("<html lang=\"de\">\n<head>\n<meta charset=\"utf-8\">\n<title>CV - Manual approach results</title>\n<link rel=\"stylesheet\" href=\"assets/jquery.dataTables.min.css\">\n<link rel=\"stylesheet\" href=\"assets/style.css\">\n<script src=\"assets/jquery-3.3.1.js\"></script>\n<script src=\"assets/jquery.dataTables.min.js\"></script>\n</head>\n")

html.write("<body>\n<div id='st'></div>\n\n<h1><a href='Results.html'>CV - Manual approach results</a></h1>\n\n<table id='t' class='display'>\n<thead>\n<tr>\n<th id='f' title='Reset index'><button id='b_index'>Index</button><br><button id='b_winner'>\"Winner\"</button></th><th title='Field #1'>Distance measure</th><th title='Field #2'>Neighbour count</th><th title='Field #3'>Descriptor 1</th><th title='Field #4'>Descriptor 2</th><th title='Field #5'>Weight</th><th title='Field #6'>Bin count</th><th title='Field #7'>Guessing accuracy (%)</th><th title='Field #8'>Time needed (min)</th><th title='Field #9'>Image set</th><th title='Run Nr.' class='rn'>Run .. to ..</th>\n</tr>\n</thead>\n<tbody>\n")
           
for i, line in enumerate(csv):
    if not(line.startswith("#") or line.isspace() or i == 0): # comment-line in csv file also starts with #
        row = line.split(";")
        
        run_nr = row[0]
        distance_measure = row[1]
        neighbour_count = row[2]
        descriptor_1 = row[3]
        descriptor_2 = row[4]
        weight = row[5]
        bin_count = row[6]
        guessing_accuracy = row[7]
        time_needed = row[8]
        image_set = row[9]
        
        html.write("<tr>")
        html.write("<td class='i'></td>")
        html.write("<td>%s</td>" % distance_measure)
        html.write("<td>%s</td>" % neighbour_count)
        html.write("<td>%s</td>" % descriptor_1)
        html.write("<td>%s</td>" % descriptor_2)
        html.write("<td>%s</td>" % weight)
        html.write("<td>%s</td>" % bin_count)
        html.write("<td class='ga'>%s</td>" % guessing_accuracy)
        html.write("<td class='tn'>%s</td>" % time_needed)
        html.write("<td>%s</td>" % image_set)
        html.write("<td class='rn'>%s</td>" % run_nr)
        html.write("</tr>\n")

html.write("</tbody>\n</table>\n\n<script src=\"assets/script.js\"></script>\n\n</body>\n</html>")

html.close()

print("\nDone!")
