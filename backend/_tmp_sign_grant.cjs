/* TEMP test helper: sign a copilot grant for audit_file 56, exactly like
 * 1audit-be's signGrant (HS256, scope 'copilot', 15m). Prints {org,user,grant}.
 * Auto-detects the Sequelize table names. Not committed; local testing only. */
const fs = require('fs');
const jwt = require('jsonwebtoken');
const mysql = require('mysql2/promise');

const BE = '/Users/susovon/live-project/1audit/1audit-be-v3';
const env = {};
for (const line of fs.readFileSync(BE + '/.env', 'utf8').split('\n')) {
  const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/i);
  if (m) {
    let v = m[2];
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1, -1);
    env[m[1]] = v;
  }
}
const secret = env.COPILOT_GRANT_SECRET || env.JWT_LOGIN_SECRET;
if (!secret) { console.error('NO SECRET'); process.exit(1); }
const DB = env.MYSQL_DATABASE;

(async () => {
  const conn = await mysql.createConnection({
    host: env.MYSQL_WRITE_HOST || env.MYSQL_READ_HOST || '127.0.0.1',
    port: Number(env.MYSQL_PORT || 3306),
    user: env.MYSQL_USER,
    password: env.MYSQL_PASSWORD,
    database: DB,
  });

  const af = 'aud_audit_files';

  const [rows] = await conn.execute(`SELECT * FROM \`${af}\` WHERE id = 56 LIMIT 1`);
  if (!rows.length) { console.error('id 56 not in ' + af); process.exit(1); }
  const f = rows[0];
  const org = f.organization_id ?? f.organizationId;
  let user = f.created_by ?? f.createdBy ?? f.created_by_id ?? f.user_id ?? f.userId ?? f.owner_id ?? f.updated_by ?? null;

  await conn.end();
  if (org == null) { console.error('no org col; keys=' + JSON.stringify(Object.keys(f))); process.exit(1); }
  if (user == null) user = Number(env.PRIME_USER_ID) || 1; // harmless fallback; reads scope by org+file
  const grant = jwt.sign(
    { scope: 'copilot', user_id: Number(user), organization_id: Number(org), audit_file_id: 56 },
    secret,
    { expiresIn: '15m', algorithm: 'HS256' },
  );
  console.log(JSON.stringify({ table: af, org, user, grant }));
})().catch((e) => { console.error('ERR', e.message); process.exit(1); });
