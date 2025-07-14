const {
  getProducts,
  getSingleProduct,
} = require("../controller/productController");
const { order } = require("../controller/orderController");
const express = require("express");

const router = express.Router();

router.get("/products", getProducts);
router.post("/orders", order);
module.exports = router;
