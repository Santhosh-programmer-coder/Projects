const express = require("express");
const dotenv = require("dotenv");
const authroute = require("./routes/auth.route");
const connectdb = require("./db/connectdb");
const cors = require("cors");
dotenv.config();
const port = process.env.PORT;

const app = express();
app.use(express.json());
app.use(cors());
app.use("/api/auth", authroute);
console.log(port);
app.listen(port, () => {
  console.log(`server is running on ${port}`);
  connectdb();
});
