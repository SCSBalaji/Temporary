const axios = require("axios");
async function test() {
  try {
    const loginRes = await axios.post("https://api4.thetvdb.com/v4/login", {
      apikey: "abbe159d-68d7-4708-8a93-cb5ebe51557f"
    });
    const token = loginRes.data.data.token;
    console.log("Token:", token.substring(0, 10) + "...");
    
    const searchRes = await axios.get("https://api4.thetvdb.com/v4/search?query=Inception&type=movie", {
      headers: { Authorization: `Bearer ${token}` }
    });
    console.log("Search result:", JSON.stringify(searchRes.data.data[0], null, 2));
  } catch (e) {
    console.error(e.response ? e.response.data : e.message);
  }
}
test();
