# -*- coding: utf-8 -*-

csv = open("Results2.csv","r")
html = open("table_for_viewing_result/index2.html", "w")

html.write("<html lang=\"de\"><head><meta charset=\"utf-8\"><title>CV - Manueller Ansatz</title><style>body{font-family:sans-serif}thead th{text-align:left;padding:10 8 !important}td{border-left:1px solid #ccc}td:first-child{border-left:none}table.dataTable.stripe tbody tr.odd, table.dataTable.display tbody tr.odd{background-color:#fff !important}table.dataTable.stripe tbody tr.even, table.dataTable.display tbody tr.even{background-color:#ddd !important}table.dataTable.display tbody tr.odd > .sorting_1, table.dataTable.order-column.stripe tbody tr.odd>.sorting_1{background-color:#daffd9 !important}table.dataTable.display tbody tr.even > .sorting_1, table.dataTable.order-column.stripe tbody tr.even>.sorting_1{background-color:#a6eda4 !important}table.dataTable.hover tbody tr:hover, table.dataTable.display tbody tr:hover, table.dataTable.display tbody tr:hover > .sorting_1, table.dataTable.order-column.hover tbody tr:hover>.sorting_1{background-color:#b8bfff !important}</style><link rel=\"stylesheet\" href=\"assets/jquery.dataTables.min.css\"> <script src=\"assets/jquery-3.3.1.js\"></script> <script src=\"assets/jquery.dataTables.min.js\"></script> </head>")
html.write("<body><h1>Manual approach results</h1><table id='t' class='display'><thead><tr><th title='Field #1'>distance_measure</th><th title='Field #2'>neighbour_count</th><th title='Field #3'>descriptor_1</th><th title='Field #4'>descriptor_2</th><th title='Field #5'>weight</th><th title='Field #6'>bin_count</th><th title='Field #7'>guessing_accuracy (%)</th><th title='Field #8'>time_needed (min)</th><th title='Field #9'>image_set</th></tr></thead><tbody>")
           
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
        html.write("</tr>")

html.write("<tfoot><tr><th title='Field #1'>distance_measure</th><th title='Field #2'>neighbour_count</th><th title='Field #3'>descriptor_1</th><th title='Field #4'>descriptor_2</th><th title='Field #5'>weight</th><th title='Field #6'>bin_count</th><th title='Field #7'>guessing_accuracy (%)</th><th title='Field #8'>time_needed (min)</th><th title='Field #9'>image_set</th></tr></tfoot></table>")
html.write("<script>$(document).ready(function(){$(\"#t\").DataTable({lengthMenu:[\"9999\",\"100\",\"50\",\"10\"],initComplete:function(){this.api().columns().every(function(){var e=this,n=$('<select><option value=\"\"></option></select>').appendTo($(e.footer()).empty()).on(\"change\",function(){var n=$.fn.dataTable.util.escapeRegex($(this).val());e.search(n?\"^\"+n+\"$\":\"\",!0,!1).draw()});e.data().unique().sort().each(function(e,t){n.append('<option value=\"'+e+'\">'+e+\"</option>\")})})}})})</script></body></html>")
html.close()

print("\nDone!")