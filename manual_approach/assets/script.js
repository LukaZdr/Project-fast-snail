$(document).ready(function() {	
    t = $('#t').DataTable({
        fixedHeader: true,
        "lengthMenu": ["9999"],
		"columnDefs": [{
            "searchable": false,
            "orderable": false,
			"width": "1%",
            "targets": 0
        }],
        // "order": [[ 7, 'desc' ]],
        initComplete: function() {
            var table = this;
            table.api().columns().every(function() {
                var column = this;
                var select = $('<br><select><option value=""></option></select>').appendTo($(column.header())).on('change', function() {
                    var val = $(this).val();
                    column.search(val ? '^' + val + '$' : '', true, false).draw();
                });
                column.data().unique().sort().each(function(value, index) {
                    select.append('<option value="' + value + '">' + value + '</option>')
                });
                $(select).click(function(e) {
                    e.stopPropagation();
                });
            });
        }
    });
	
	t.column(0).nodes().each(function (cell, i) {
		cell.innerHTML = i+1;
	});
		
	$('#b_index').click(function() {
		t.column(0).nodes().each(function (cell, i) {
			cell.innerHTML = i+1;
		});
	});
	
	$('#b_winner').click(function() {
		var activeColumn = t.order();
		let tempContent = "";
		let tempIndex = 0;
		t.column(activeColumn[0][0], {search:'applied', order:'applied'}).nodes().each(function (cell, i) {
			if(cell.innerHTML !== tempContent) {
				tempIndex = i+1;
				tempContent = cell.innerHTML;
			}
			t.column(0, {search:'applied', order:'applied'}).nodes()[i].innerHTML = tempIndex;
		});
	});
	
	let timeTotal = 0.0;
	$('.tn').each(function(){
		timeTotal += parseFloat(this.innerHTML);
	});
	console.log("Berechnungszeit gesamt in Tagen: " + timeTotal / 60 / 24);
});
