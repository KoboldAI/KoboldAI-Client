// Global Definitions
var fav_icon2 = "data:image/x-icon;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAB+1BMVEUAAAAAAAAAAAAAAAAAAQAAAAAAAQAAAAAAAAASFhBBWD4iUyoFEwgFEwguUTM+VDoMFAwAAAA+elIudz8AAAAAAAA0MigyLyQAAAAbLh1LdElSbUoVMBkAAABAZ0M2fkUAAAABAQFMiGQraDkAAQANFxEGFQkLFg8EEAYAAAAsZDonZjUAAABCgVVAnFYrSjhEjFpFi1sdRScAAAAjOi8VMxx1dGOFgGYAAABOTEabmIdlYlQaGhgaGhddXFauqY5JRjoAAAAAAAABAQFGeExIl1lX0XRW0XRHi1RFe02vv5W31KFd1Hpc1Hpe1HvO1KvDvJlqZ1plYVOmoIVt1IFl1H7AuZp1cV9jX1AmSCw3Nzg7NmA1MTJuz4Bm1H5MST9HPl9BQEMgNiNXgWKiobFgXICDd5dfw3RZVnJiV3zGv9Bqf29Oj2G/v8hTTpGhl8dbxHVd0npiYoxhWJvIxtlcimZFn1lRclg9SkZNblZBeEpDbEZCa0ZBc0hLY1BAS1BdaV87j01Vx3FWynJSrGZOhlVasGtas2xatm1at21WnWJQm15WyXJQvmlavnBZrGlEYEJWe1RBWz9Um2BavXBgxn9XhllGY0RLaklXiFlTwG5OpmVSfFNMbUpGZEVLa0lShldEhVCChHiKiHvWz6/Kw6WWlZGAfmj///8kr0X+AAAARHRSTlMAASFrcAhxIjLb/vWvsPb+20b4+DFFyMkz2vf43CP9/m5y9vZysLGvsQn19mz+/tz4+NxHycr3+Ejb/vaxsPX+3TRtcBrzrrgAAAABYktHRKhQCDaSAAAAB3RJTUUH5gYJFyQy3tftxgAAAQBJREFUGNNjYGBgYGRiZmFlZWNmZ2SAAA5OLm4eXj5+AQ6ogKCQi6ubu4ensCCIxygiKubl7ePr6+cfIC4owcjAJCkVGBQc4usbGhYeIS0jy8AsFxkVHRPr6xsXn5CYJK/AoKiUnJKalg5UkZGZla2swsCqmpObl1/g61tYVFxSqsbKwKpeVl5RWVVdU1tX39CoocnAotXU3NLa1t7R2dXd06utwqCj6+vb1z9h4sRJk6f4+uopMLDrG0z1nTZ94sQZM31nGRrJMjBKGJvMnjN3wrz5CxaaCnKAvSNqtmjxkqXLlptbQP0iYmllbWNrZ+/gCBVgZHdS1GR1VpAFqQcApI0/jqlZOvEAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjItMDYtMDlUMjM6MzY6NTArMDA6MDDi0xr+AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDIyLTA2LTA5VDIzOjM2OjUwKzAwOjAwk46iQgAAAABJRU5ErkJggg==";
var fav_icon1 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAB+FBMVEUAAAAAAAAAAAAAAAAAAAEAAAAAAQEAAAAAAAAUFRlLVGYrSWgHEBoHEBk3S19HUGMOExkAAABOcos7apIAAAAAAAA2Ly01KyoAAAAgKzdVaX9bZHIaKzwAAABKYHhDcZgAAAABAQFfgJY2XX0AAQEQFhoIEhwOFRgGDRUAAAAAAQE3W3cyWnwAAABSeJJRjLs1R1FVgaFWgJ4lPlMAAAAsOD4aLj55bm2Md3QAAABPSkmfko9pXlsbGRkbGRlfWlm1oJxMQkAAAAAAAAABAQFTb4tYibFtvPpWgKNScpC6s7nExtNzwPp1wPnZx8jMsKtuZGFoXVutmJODwfJ7wfbHr6p5a2hnW1gtQlI4ODk7N2A2LzWDvet8wPZPRkRHPl9CQUQlMTthe4+ko7RhXYGEeJhzsuJaVXRjWHzIwtNwfYddhqLCwcpTTpGimMhvsuVzv/djYpBgWJvLydxlgptVirdZbX1ASFZUaXtOb4xOZX1OZHxNa4ZRX21DSV5gaG9Je6lqsepstO1knclcfJxtoc5tpNFuptVup9ZnkbdgjrVss+xjpuBvrd9snspOW29jdI5LVmlkj7Vvrd54t+RlfptQXXJWZHtlf51oruNgmMFfdJBYZn1RXnRWZXthfZxSeZiGgYGOhYLdxb/RubWZlpWFd3T////2kwjgAAAARXRSTlMAASFrcAhxIjLb/vWvsPb+20b4+DFFyMkz2vf43CP9/m5y9vZysLGvsQlw9fZs/v7c+PjcR8nK9/hI2/72sbD1/t00bXBAFktiAAAAAWJLR0SnwLcrAwAAAAd0SU1FB+YGCRchHQhxJNoAAAD/SURBVBjTY2BgYGBkYmZhZWVjZmdkgAAOTi5uHl4+fgEOqICgkKubu7uHp7AgiMcoIirm5e3j4+Pr5y8uKMHIwCQpFRAYFOzjExIaFi4tI8vALBcRGRUd4+MTGxefkCivwKColJSckpoGVJGekZmlrMLAqpqdk5uX7+NTUFhUXKLGysCqXlpWXlFZVV1TW1ffoKHJoKXd2NTc0trW3tHZ1d2jo8Kgq+fj09vXP2HCxEmTfXz0FRjYDQyn+EydNmHC9Bk+M42MZRkYJUxMZ82e0z933vwFZoIcYO+Imi9ctHjJ0mUWllC/iFhZ29ja2Ts4OkEFGNmdFTVZXRRkQeoBhkE/Yj5NSZ4AAAAldEVYdGRhdGU6Y3JlYXRlADIwMjItMDYtMDlUMjM6MzM6MjgrMDA6MDA90JbEAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDIyLTA2LTA5VDIzOjMzOjI4KzAwOjAwTI0ueAAAAABJRU5ErkJggg==";
var fav_icon = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAB8lBMVEUAAAAAAAAAAAAAAAABAAAAAAABAAAAAAAAAAAdEBB0Pz5rKCgaBwcZBwdkMzJxPDocDAwAAACLTU6SOzsAAAAAAAA9Mic/LyEAAAA6HByQUUaIVEY+GBgAAACAQkKaQUIAAAABAQGWXl9+NjYBAAAaEBAcCAgZDQ0WBQUAAAB3Nzd9MjIAAACTUVK7UVJRNTWhVVaeVldTJSUAAAA+LC0+GhuGcmCgf2EAAABUTESrl4NzYlEdGhcdGhdiXFbIqIhWRjcAAAAAAAABAQGUSkq1VVX6bW6oUVGXS0vmro7+uJn6c3T6dXX/yqPnu5F3aFhxYVG/oH/7gHv6enjeuJOEcFtzX01VLCs4ODk7NmA5MTH1gHr6e3hWSTxHPl9CQUQ/JCKPYGGko7RhXYGEeJjmcW9cVnFjWH3IwtOHb3CjXV3CwcpTTpGimMjlb3D4c3RmYI1gWJvLydybZWW+T0x+V1hRP0Z7U1WTSEiHRUWGRUSORkZuTlBRQVBwX2CvRkXtaGjvamrNYWKmU1PVZ2fXaGjbaWncaWnAX1+7W1vkYF/ja2zRZWV9QkGeVFN2Pz69XV3ia2zkeHmpWFd/REOJSUirWVjjaGjBYGCeUlKMSkl8QkGBRUSoVlWeUE2QgXeWiHr1zqjmw5+bl5KVe2T///8NZLRGAAAARHRSTlMAASFrcAhxIjLb/vWvsPb+20b4+DFFyMkz2vf43CP9/m5y9vZysLGvsQn19mz+/tz4+NxHycr3+Ejb/vaxsPX+3TRtcBrzrrgAAAABYktHRKUuuUovAAAAB3RJTUUH5gYJFzsfVlK/LQAAAP9JREFUGNNjYGBgYGRiZmFlZWNmZ2SAAA5OLm4eXj5+AQ6ogKCQi6ubm7uHsCCIxygiKubp5e3t7ePrJy4owcjAJCnlHxAY5O0dHBIaJi0jy8AsFx4RGRXt7R0TGxefIK/AoKiUmJSckgpUkZaekamswsCqmpWdk5vn7Z1fUFhUrMbKwKpeUlpWXlFZVV1TW1evocnAotXQ2NTc0trW3tHZ2KWtwqCj6+3d3dPb19c/YaK3t54CA7u+wSTvyVP6+qZO855uaCTLwChhbDJj5qzZc6bOnWcqyAH2jqjZ/AULFy1eYm4B9YuIpZW1ja2dvYMjVICR3UlRk9VZQRakHgAlRz6K4dvoSgAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wNi0wOVQyMzo1OTozMSswMDowMJt1iQMAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDYtMDlUMjM6NTk6MzErMDA6MDDqKDG/AAAAAElFTkSuQmCC"
var submit_start;

var favicon = {

	// Change the Page Icon and Title.
	change: function(iconURL) {
		this.addLink(iconURL, "icon");
		this.addLink(iconURL, "shortcut icon");
	},

	addLink: function(iconURL, relValue) {
		var link = document.createElement("link");
		link.type = "image/x-icon";
		link.rel = relValue;
		link.href = iconURL;
		this.removeLink(relValue);
		this.docHead.appendChild(link);
	},

	removeLink: function(relValue) {
		var links = this.docHead.getElementsByTagName("link");
		for (var i = 0; i < links.length; i++) {
			var link = links[i];
			if (link.type == "image/x-icon" && link.rel == relValue) {
				this.docHead.removeChild(link);
				return; // Assuming only one match at most.
			}
		}
	},
	
	swapLink: function() {
		if (this.run == true) {
			if (this.icon == 1) {
				this.change(fav_icon2);
				this.icon = 2;
			} else {
				this.change(fav_icon1);
				this.icon = 1;
			}
		}
	},
	
	auto_swap: function() {
		if (this.run == true) {
			this.swapLink();
			setTimeout(() => {  this.auto_swap(); }, 1000);
		}
	},
	
	start_swap: function() {
		this.run = true;
		this.auto_swap();
		submit_start = Date.now();
	},
	
	stop_swap: function() {
		this.run = false;
		this.change(fav_icon);
		if (typeof submit_start !== 'undefined') {
			$("#runtime")[0].innerHTML = `Execution time: ${Math.round((Date.now() - submit_start)/1000)} sec`;
			delete submit_start;
		}
	},
	
	docHead:document.getElementsByTagName("head")[0]
}