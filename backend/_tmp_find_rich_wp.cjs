const fs=require('fs'),mysql=require('mysql2/promise');
const BE='/Users/susovon/live-project/1audit/1audit-be-v3';const env={};
for(const l of fs.readFileSync(BE+'/.env','utf8').split('\n')){const m=l.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/i);if(m){let v=m[2];if((v.startsWith('"')&&v.endsWith('"'))||(v.startsWith("'")&&v.endsWith("'")))v=v.slice(1,-1);env[m[1]]=v;}}
(async()=>{const c=await mysql.createConnection({host:env.MYSQL_WRITE_HOST||env.MYSQL_READ_HOST,port:Number(env.MYSQL_PORT||3306),user:env.MYSQL_USER,password:env.MYSQL_PASSWORD,database:env.MYSQL_DATABASE});
const [gc]=await c.execute("SELECT COLUMN_NAME cn FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=? AND TABLE_NAME='aud_wp_general_sections'",[env.MYSQL_DATABASE]);
console.log('general_sections cols:',gc.map(x=>x.cn).join(','));
const [rows]=await c.execute(`
  SELECT g.audit_file_id af, g.working_paper_id wp, g.organization_id org, COUNT(*) procs
  FROM aud_wp_programs_checklists_sections p
  JOIN aud_wp_general_sections g ON g.id = p.section_id
  WHERE p.\`procedure\` IS NOT NULL AND CHAR_LENGTH(p.\`procedure\`)>120 AND p.status='active' AND g.status='active'
  GROUP BY g.audit_file_id, g.working_paper_id, g.organization_id
  ORDER BY procs DESC LIMIT 8`);
console.log('rich WPs (af/wp/org/procs):');
rows.forEach(r=>console.log(`  af=${r.af} wp=${r.wp} org=${r.org} procs=${r.procs}`));
await c.end();})().catch(e=>{console.error('ERR',e.message);process.exit(1);});
