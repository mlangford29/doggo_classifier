#breed_message = what message to display
#photo = (put inside of an <img src = {photo}/>)
#name = name of nearest dog
#age = age of dog
#sex = sex of dog
#descript = descript of dog

#Here is the nearest dog of that breed looking for a home[at the first <p>]

disp_str = """<! DOCTYPE html>	
<html lang = "en" id = "test">
	<head>
		<title>Dog View</title>
		<meta charset = "Utf-8"/>
		
		<style>
#test{{
	color: #0F1C2D;
	background-color: #7AA1D3;
	font-family: Verdana, Geneva, sans-serif;
}}
#doggo{{
	background-color: #E29CA4;
	padding: 20px;
	border-radius: 20px;
	text-align: center;

}}
#c1{{
	text-align:center;
}}
#p1{{
	position: relative;
	top: 30%;
	left: 40%;
}}
#c2{{
	position: relative;
	top: 40%;
	left: 30%;
}}
#d{{
	position: relative;
	top: 70%;
	left: 20%;
}}
	</style>
	</head>
	
	<body>
	<h1 id = "doggo">Dog-go</h1>
	<h1>{}</h1>
	<p>Here is the nearest dog of that breed looking for a home</p>
	<img src = "{}" alt = "good doggo" width = "15%" id = "doggo"/>
	<h2>Meet {}! Age: {}</h2>
	{}
	<p>Description:</p>
	{}
	</body>

</html>"""