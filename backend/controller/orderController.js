const orderModel = require("../model/orderModel");
const order = async (req, res) => {
  console.log(req.body, "DATA"); // 👈 This should log the object

  res.json({
    success: true,
    message: "Order placed",
  });
};
module.exports = { order };
