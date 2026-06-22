const fs=require('fs'),mysql=require('mysql2/promise');
const BE='/Users/susovon/live-project/1audit/1audit-be-v3';const env={};
for(const l of fs.readFileSync(BE+'/.env','utf8').split('\n')){const m=l.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/i);if(m){let v=m[2];if((v.startsWith('"')&&v.endsWith('"'))||(v.startsWith("'")&&v.endsWith("'")))v=v.slice(1,-1);env[m[1]]=v;}}
(async()=>{const c=await mysql.createConnection({host:env.MYSQL_WRITE_HOST||env.MYSQL_READ_HOST,port:Number(env.MYSQL_PORT||3306),user:env.MYSQL_USER,password:env.MYSQL_PASSWORD,database:env.MYSQL_DATABASE});
const [r]=await c.execute("SELECT section_type, COUNT(*) n FROM aud_wp_general_sections WHERE working_paper_id=1184 AND status='active' GROUP BY section_type");
console.log('WP1184 section_type counts:',JSON.stringify(r));
const [p]=await c.execute("SELECT g.section_type st, COUNT(*) n FROM aud_wp_programs_checklists_sections p JOIN aud_wp_general_sections g ON g.id=p.section_id WHERE g.working_paper_id=1184 AND p.status='active' GROUP BY g.section_type");
console.log('WP1184 PROGRAM-detail section_type:',JSON.stringify(p));
await c.end();})().catch(e=>{console.error('ERR',e.message);process.exit(1);});
