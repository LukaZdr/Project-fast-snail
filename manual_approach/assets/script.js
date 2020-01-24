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
        "order": [[ 1, 'asc' ]],
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
	
	$('#new_index').click(function() {
		t.column(0, {search:'applied', order:'applied'}).nodes().each( function (cell, i) {
			cell.innerHTML = i+1;
		});
	});
});
