var submitb;
window.onload = function(){
	submitb = document.getElementById("formfile");
	submitb.onsubmit = function(){
		if(submitb.dogspot.value == ""){
			window.alert("Please Give Me a File")
			return(false)
	}
}}
