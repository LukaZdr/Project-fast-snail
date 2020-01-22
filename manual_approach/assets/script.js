$(function(){
		setTimeout(function() {
			$(window).scroll(function() {
				if ($(this).scrollTop() >= 100) {
					$('thead').addClass('stickyNav');
				}
				if ($(this).scrollTop() <= 10) {
					$('thead').removeClass('stickyNav');
				}
			});
		}, 200);
	});