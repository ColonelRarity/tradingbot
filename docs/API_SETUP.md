# Як підключити Binance Futures API (2025–2026)

Binance змінив подачу: тепер є окрема сторінка **Binance API** з розділами Spot / Derivatives, а ключі як і раніше створюються в **API Management** — але шлях до неї може бути через новий хаб.

## 1. Відкрийте API Management

Будь-який з варіантів (залежить від регіону / домену):

| Шлях | URL (приклад) |
|------|----------------|
| Новий хаб API | `https://www.binance.com/en/binance-api` (у вас може бути `binance.bh`) |
| Класичний | Профіль → **Account** → **API Management** |
| Пряме посилання | `https://www.binance.com/en/my/settings/api-management` |

На сторінці зі скріншота натисніть блок **Create / Manage API Key** (діаграма Trading Connectivity) — вас перекине в API Management.

**Перед створенням:** увімкнений 2FA, пройдений KYC, акаунт активований (був депозит).

## 2. Create API

1. **Create API** (або **Create API Key**).
2. Тип ключа — оберіть **System generated** (HMAC).  
   - Це дає **API Key** + **Secret Key** (Secret показують **один раз** — збережіть одразу).  
   - **Self-generated** (Ed25519 / RSA) — для цього бота **не підходить** без переписування підпису.
3. Мітка (label), наприклад: `scalper-bot-live`.
4. Підтвердіть 2FA / passkey.

## 3. Дозволи (Permissions)

Для нашого скальпера на **USDⓈ-M Futures**:

| Дозвіл | Потрібно |
|--------|----------|
| Enable Reading | Так |
| Enable Futures / USDⓈ-M Futures | Так |
| Enable Spot & Margin Trading | Ні (якщо бот тільки futures) |
| Enable Withdrawals | **НІ** |
| Enable Universal Transfer | Ні |

Рекомендовано: **Restrict access to trusted IPs only** — додайте свій публічний IP (з `https://ifconfig.me`).

## 4. Futures акаунт

Окремо в терміналі Binance:

1. **Derivatives** → **USDⓈ-M Futures** — акаунт має бути відкритий.
2. Переконайтесь, що на Futures є USDT (ви вже поповнили — ок).
3. **Position mode**: One-way або Hedge — бот підтримує обидва (`POSITION_MODE` у `.env`).

## 5. Куди вставити ключі в боті

Скопіюйте **`.env.example`** → **`.env`** (не `env.example.ps1`!).

Файл `.env` — це простий текст `KEY=value`, **без** `$env:` і **без** `Write-Host`:

```env
BINANCE_API_KEY=ваш_api_key
BINANCE_API_SECRET=ваш_secret_key
BINANCE_FUTURES_BASE_URL=https://fapi.binance.com
CONFIRM_LIVE_TRADING=true
```

Якщо зручніше PowerShell — використовуйте окремий **`env.local.ps1`** (див. `env.example.ps1`) і запускайте `. .\env.local.ps1` перед `python main.py`. Не змішуйте синтаксис PowerShell у файлі `.env`.

Перевірка без торгівлі:

```bash
python verify_setup.py
```

## 6. Demo / Testnet (не плутати з live)

| Що | URL |
|----|-----|
| **Реальна торгівля** | `https://fapi.binance.com` |
| **Futures testnet (API)** | `https://demo-fapi.binance.com` (актуально в офіційній документації) |

«Demo Trading» у веб-інтерфейсі Binance і **API testnet** — різні речі. Для бота потрібні саме **API keys** з API Management, прив’язані до того середовища, яке ви вказали в `BINANCE_FUTURES_BASE_URL`.

## 7. Типові помилки

| Помилка | Причина |
|---------|---------|
| `-2015` Invalid API-key | Невірний ключ або ключ не для Futures |
| `-1022` Signature failed | Помилка в Secret або годинник ПК (синхронізуйте час Windows) |
| IP restricted | IP не в whitelist — додайте IP або тимчасово зніміть обмеження |
| Ed25519 key | Обрано self-generated ключ — створіть новий **HMAC** ключ |

## 8. Зв’язок з кодом бота

Бот використовує **HMAC** через `binance-futures-connector` (`UMFutures`), endpoint `fapi.binance.com`.  
Приклад на сайті Binance з `from binance.futures import Futures` — інша обгортка, але **ті самі ключі** і той самий REST API.
