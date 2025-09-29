import { useState } from "react";
import axios from "axios";

export default function useApi(url, method = "get") {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function callApi(data = null, config = {}) {
    setLoading(true);
    setError(null);
    try {
      const res = await axios({ url, method, data, ...config });
      return res.data;
    } catch (err) {
      setError(err.response?.data?.error || err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }

  return { callApi, loading, error };
}