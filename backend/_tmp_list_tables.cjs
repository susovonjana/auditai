const fs=require('fs'),mysql=require('mysql2/promise');
const BE='/Users/susovon/live-project/1audit/1audit-be-v3';const env={};
for(const l of fs.readFileSync(BE+'/.env','utf8').split('\n')){const m=l.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/i);if(m){let v=m[2];if((v.startsWith('"')&&v.endsWith('"'))||(v.startsWith("'")&&v.endsWith("'")))v=v.slice(1,-1);env[m[1]]=v;}}
const DB=env.MYSQL_DATABASE;
(async()=>{const c=await mysql.createConnection({host:env.MYSQL_WRITE_HOST||env.MYSQL_READ_HOST,port:Number(env.MYSQL_PORT||3306),user:env.MYSQL_USER,password:env.MYSQL_PASSWORD,database:DB});
const [t]=await c.execute("SELECT TABLE_NAME n FROM information_schema.TABLES WHERE TABLE_SCHEMA=? AND TABLE_NAME LIKE '%file%' ORDER BY TABLE_NAME",[DB]);
console.log('FILE tables:',t.map(r=>r.n).join(', '));
// which of these has a row id=56 and an organization_id column
for(const r of t){const n=r.n;try{const [cols]=await c.execute("SELECT COLUMN_NAME cn FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=? AND TABLE_NAME=?",[DB,n]);const cn=cols.map(x=>x.cn);if(cn.includes('organization_id')&&cn.includes('id')){const [rows]=await c.execute(`SELECT id,organization_id FROM \`${n}\` WHERE id=56 LIMIT 1`);if(rows.length){console.log('  -> id56 IN',n,'org=',rows[0].organization_id,'| userish cols:',cn.filter(x=>/(created_by|user|owner|updated_by)/i.test(x)).join(','));}}}catch(e){}}
await c.end();})().catch(e=>{console.error('ERR',e.message);process.exit(1);});
