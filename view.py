#breed_message = what message to display
#photo = (put inside of an <img src = {photo}/>)
#name = name of nearest dog
#age = age of dog
#sex = sex of dog
#descript = descript of dog

#Here is the nearest dog of that breed looking for a home[at the first <p>]

html = """<! DOCTYPE html>	
<html lang = "en">
	<head>
		<title>Dog View</title>
		<meta charset = "Utf-8"/>
	</head>
	
	<body>
	<h1>{}</h1>
	<p>Here is the nearest dog of that breed looking for a home</p>
	<img src = "{}" alt = "good doggo"/>
	<h2>Meet {}! Age: {}</h2>
	{}
	<p>Description:</p>
	{}
	</body>
</html>"""