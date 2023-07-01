Updater = function(image_id, timeout, show_invisible){
    this.img = $(image_id);
    this.timeout = timeout;
    this.source = $(image_id).attr('src');
    this.show_invisible = show_invisible || false;

    this.timer = 0;

    this.img.on('load', $.proxy(this.run, this));
    this.img.on('error', $.proxy(this.run, this));

    this.run();
}

Updater.prototype.update = function(){
    if(this.img.is(":visible") || this.show_invisible){
        if(this.source.indexOf("?") > 0)
            this.img.attr('src', this.source + '&rnd=' + Math.random());
        else
            this.img.attr('src', this.source + '?rnd=' + Math.random());
    } else
        this.run();
}

Updater.prototype.run = function(){
    clearTimeout(this.timer);
    this.timer = setTimeout($.proxy(this.update, this), this.timeout);
}
