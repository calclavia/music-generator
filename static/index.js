document.addEventListener('DOMContentLoaded', function(event) {
    var slider_baroque = document.getElementById('baroque');
    var slider_classical = document.getElementById('classical');
    var slider_romantic = document.getElementById('romantic');
    var slider_modern = document.getElementById('modern');
    var baroque = slider_baroque.value / 100;
    var classical = slider_classical.value / 100;
    var romantic = slider_romantic.value / 100;
    var modern = slider_modern.value / 100;
    var stream_url = '/stream.wav?';

    function build_url(baroque, classical, romantic, modern) {
        return stream_url + 'baroque=' + baroque + '&classical=' + classical + '&romantic=' + romantic + '&modern=' + modern;
    }

    slider_baroque.onchange = function() {
        baroque = this.value / 100;
        var url = build_url(baroque, classical, romantic, modern);
        $.get(url, function(data) {
            console.log('Sent GET request to ' + url);
        });
    };

    slider_classical.onchange = function() {
        classical = this.value / 100;
        var url = build_url(baroque, classical, romantic, modern);
        $.get(url, function(data) {
            console.log('Sent GET request to ' + url);
        });
    };

    slider_romantic.onchange = function() {
        romantic = this.value / 100;
        var url = build_url(baroque, classical, romantic, modern);
        $.get(url, function(data) {
            console.log('Sent GET request to ' + url);
        });
    };

    slider_modern.onchange = function() {
        modern = this.value / 100;
        var url = build_url(baroque, classical, romantic, modern);
        $.get(url, function(data) {
            console.log('Sent GET request to ' + url);
        });
    };
});
