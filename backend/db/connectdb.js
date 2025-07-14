const mongoose = require("mongoose");

const connectdb = async () => {
  try {
    await mongoose.connect(process.env.MANGO_URL);
    console.log("Database Connected");
  } catch (error) {
    console.log(`Error in Database connection ${error}`);
  }
};
module.exports = connectdb;
