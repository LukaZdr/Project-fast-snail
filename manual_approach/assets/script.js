$(document).ready(function() {
	// Init table
    t = $("#t").DataTable({
        fixedHeader: true,
        lengthMenu: ["999"],
		columnDefs: [{
            searchable: false,
            orderable: false,
			width: "50px",
            targets: 0
			},
			{
			type: 'natural', 
			targets: '_all'
		}],
		order: [[0, 'asc']],
		drawCallback: function(settings) {
			writeTime();
			colorMap();
		},
        initComplete: function() {
            let table = this;
			// Set up filter for column 0-9
            table.api().columns([0,1,2,3,4,5,6,7,8,9]).every(function() {
                let column = this;
                let select = $("<br><select><option value=''></option></select>").appendTo($(column.header())).on("change", function() {
                    let val = $(this).val();
                    column.search(val ? "^" + val + "$" : "", true, false).draw();
                });
                column.data().unique().sort().each(function(value, index) {
                    select.append("<option value='" + value + "'>" + value + "</option>");
                });
                $(select).click(function(e) {
                    e.stopPropagation();
                });
				$('#b_winner').click(function(e) {
                    e.stopPropagation();
                });
				$('#b_index').click(function(e) {
                    e.stopPropagation();
                });
            });
			// Set up search for elements that are val1 or val2 or between
			table.api().columns([10]).every(function() {
                let column = this;
                let select = $("<br><select id='s1'><option value=''></option></select><select id='s2'><option value=''></option></select>").appendTo($(column.header())).on("change", function() {
                    let val1 = $('#s1').val();
					let val2 = $('#s2').val();
					let regexString = "";
                    for (let i = val1; i <= val2; i++) {
						if(i == val2) {
							regexString += "^" + i + "$";
						}
						else {
							regexString += "^" + i + "$|";
						}
					}
					// console.log("regexString: " + regexString);
					column.search(regexString, true, false ).draw();
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
		let activeColumn = t.order();
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
	
	// Colorize guessing accuracy column with green to red
	function colorMap() {
		let allGAs = $('.ga');
		let statusGreen = 80.0;
		let statusRed = 25.00;
		allGAs.each(function() {
			let val = (this.innerHTML - statusRed) / (statusGreen - statusRed) * 100; //Y=(X-A)/(B-A)*(D-C)+C
			$(this).css("background-color", "hsl(" + val + ", 70%, 60%)");
		});
	}
});
