# -*- coding: utf-8 -*-

csv = open("Results.csv","r")
html = open("Results.html", "w")

html.write("<html lang=\"de\">\n<head>\n<meta charset=\"utf-8\">\n<title>CV - Manual approach results</title>\n<link rel=\"stylesheet\" href=\"assets/style.css\">\n<link rel=\"stylesheet\" href=\"assets/jquery.dataTables.min.css\">\n<script src=\"assets/jquery-3.3.1.js\"></script>\n<script src=\"assets/jquery.dataTables.min.js\"></script>\n</head>\n")
html.write("<body>\n\n<h1><a href='Results.html'>Manual approach results</a></h1>\n\n<table id='t' class='display'>\n<thead>\n<tr>\n<th title='Field #1'>Distance measure</th><th title='Field #2'>Neighbour count</th><th title='Field #3'>Descriptor 1</th><th title='Field #4'>Descriptor 2</th><th title='Field #5'>Weight</th><th title='Field #6'>Bin count</th><th title='Field #7'>Guessing accuracy (%)</th><th title='Field #8'>Time needed (min)</th><th title='Field #9'>Image set</th>\n</tr>\n</thead>\n<tbody>\n")
           
for i, line in enumerate(csv):
    if(i != 0):
        row = line.split(";")
        
        distance_measure = row[0]
        neighbour_count = row[1]
        descriptor_1 = row[2]
        descriptor_2 = row[3]
        weight = row[4]
        bin_count = row[5]
        guessing_accuracy = row[6]
        time_needed = row[7]
        image_set = row[8]
        
        html.write("<tr>")
        html.write("<td>%s</td>" % distance_measure)
        html.write("<td>%s</td>" % neighbour_count)
        html.write("<td>%s</td>" % descriptor_1)
        html.write("<td>%s</td>" % descriptor_2)
        html.write("<td>%s</td>" % weight)
        html.write("<td>%s</td>" % bin_count)
        html.write("<td>%s</td>" % guessing_accuracy)
        html.write("<td>%s</td>" % time_needed)
        html.write("<td>%s</td>" % image_set)
        html.write("</tr>\n")

html.write("</tbody>\n</table>\n\n<script>\n$(document).ready(function(){$('#t').DataTable({fixedHeader:true,\"lengthMenu\":[\"9999\"],initComplete:function(){var table=this;table.api().columns().every(function(){var column=this;var select=$('<br><select><option value=\"\"></option></select>').appendTo($(column.header())).on('change',function(){var val=$(this).val();column.search(val?'^'+val+'$':'',true,false).draw();});column.data().unique().sort().each(function(value,index){select.append('<option value=\"'+value+'\">'+value+'</option>')});$(select).click(function(e){e.stopPropagation();});});}});});\n</script>\n</body>\n</html>")
html.close()

print("\nDone!")
