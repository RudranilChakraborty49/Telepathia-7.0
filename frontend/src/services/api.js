// services/api.js
// Single source of truth for all backend API calls

import axios from 'axios'

// Base URL — change this in one place when deploying
const API_BASE = 'http://localhost:5000/api'

// Axios instance with default settings
const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,    // 30s — batch endpoint can be slow
  headers: {
    'Content-Type': 'application/json',
  },
})

// ───── API METHODS ─────

export const apiHealth = async () => {
  const { data } = await client.get('/health')
  return data
}

export const apiGetTickers = async () => {
  const { data } = await client.get('/tickers')
  return data
}

export const apiGetSectors = async () => {
  const { data } = await client.get('/sectors')
  return data
}

export const apiGetStock = async (ticker, period = '1y') => {
  const { data } = await client.get(`/stock/${ticker}`, { params: { period } })
  return data
}

export const apiGetSignal = async (ticker) => {
  const { data } = await client.get(`/signal/${ticker}`)
  return data
}

export const apiGetAllSignals = async () => {
  const { data } = await client.get('/signals/all')
  return data
}

export const apiGetBacktest = async () => {
  const { data } = await client.get('/backtest')
  return data
}

export default {
  apiHealth,
  apiGetTickers,
  apiGetSectors,
  apiGetStock,
  apiGetSignal,
  apiGetAllSignals,
  apiGetBacktest,
}
