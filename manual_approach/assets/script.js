$(document).ready(function() {
	// Init table
    t = $("#t").DataTable({
        fixedHeader: true,
        "lengthMenu": ["999"],
		"columnDefs": [{
            "searchable": false,
            "orderable": false,
			"width": "50px",
            "targets": 0
        }],
		"drawCallback": function(settings) {
			writeTime();
			colorMap();
		},
        initComplete: function() {
            var table = this;
            table.api().columns().every(function() {
                var column = this;
                var select = $("<br><select><option value=''></option></select>").appendTo($(column.header())).on("change", function() {
                    var val = $(this).val();
                    column.search(val ? "^" + val + "$" : "", true, false).draw();
                });
                column.data().unique().sort().each(function(value, index) {
                    select.append("<option value='" + value + "'>" + value + "</option>");
                });
                $(select).click(function(e) {
                    e.stopPropagation();
                });
            });
			// Set index
			table.api().column(0, {search:"applied", order:"applied"}).nodes().each(function(cell, i) {
				cell.innerHTML = i + 1;
			});
			$("#b_index").addClass("clicked");
        }
    });
	
	// Button for indexing
	$("#b_index").click(function() {
		t.column(0, {search:"applied", order:"applied"}).nodes().each(function (cell, i) {
			cell.innerHTML = i + 1;
		});
		$("#b_winner").removeClass("clicked");
		$(this).addClass("clicked");
	});
	
	// Button for showing the "winner" (other indexing)
	$("#b_winner").click(function() {
		var activeColumn = t.order();
		let tempContent = "";
		let tempIndex = 0;
		t.column(activeColumn[0][0], {search:"applied", order:"applied"}).nodes().each(function(cell, i) {
			if(cell.innerHTML !== tempContent) {
				tempIndex = i + 1;
				tempContent = cell.innerHTML;
			}
			t.column(0, {search:"applied", order:"applied"}).nodes()[i].innerHTML = tempIndex;
		});
		$("#b_index").removeClass("clicked");
		$(this).addClass("clicked");
	});

	// Show time needed
	function writeTime() {
		let timeTotal = 0.0;
		$(".tn").each(function(){
			timeTotal += parseFloat(this.innerHTML);
		});
		let hours = timeTotal / 60;
		let days = hours / 24;
		let text = $("#t_info").text() + " | Time needed: " + hours.toFixed(2) + "h (" + days.toFixed(2) + "d)";
		$("#t_info").text(text);
	}
	
	function colorMap() {
		let allGAs = $('.ga');		
		//Get min/max
		let max = 100.0; //0.0;
		let min = 25.00; //99.9;
		// allGAs.each(function() {
			// if(this.innerHTML > max) {
				// max = this.innerHTML;
			// }
			// else if(this.innerHTML < min) {
				// min = this.innerHTML;
			// }
		// });
		allGAs.each(function() {
			let val = (this.innerHTML - min) / (max - min) * 100;  //Y = (X-A)/(B-A) * (D-C) + C     -vllt mit min = 0, max = 100? d und c= 100 
			$(this).css("background-color", "hsl(" + val + ", 70%, 60%)");
		});
	}
	
	
});
