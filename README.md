# NumberRec
##For User
----------
  input"docker stack deploy -c docker-compose.yml webcube"<br>
  to start the project.<br>
  URL:localhost<br>

##For Developer
---------------
  app.py:      Flask view file.<br>
  form:        The CNN Model.<br>
  model.py:    Model Trainer.<br>
  mnistout.py: Export MNIST data for test<br>
  recpic.py:   Test Model reading function from form<br>
  templates:   Htmls<br>
  static:      JS,CSS<br>
  data:        Cassandra and redis will save data in this dir<br>
  Dockerfile,requirements.txt :   Container Init (Docker config)<br>
  docker-compose.yml:             Config docker services (Docker config)<br>
