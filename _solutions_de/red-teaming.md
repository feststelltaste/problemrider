---
title: Red Teaming
description: Durchführung umfassender und realistischer Angriffe auf die
  eigenen Systeme.
category:
- Security
- Testing
problems:
- authentication-bypass-vulnerabilities
- authorization-flaws
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- monitoring-gaps
- insufficient-testing
- secret-management-problems
- system-outages
layout: solution
lang: de
en_slug: red-teaming
related_solutions:
- slug: security-tests-by-external-parties
  similarity: 0.75
- slug: vulnerability-scans
  similarity: 0.75
- slug: threat-modeling
  similarity: 0.75
- slug: penetration-tests
  similarity: 0.75
- slug: security-training
  similarity: 0.75
- slug: regression-tests
  similarity: 0.75
---

## Description

Red Teaming ist eine autorisierte, realistische Simulation, wie ein entschlossener Angreifer ein System tatsächlich angreifen würde, wobei Techniken wie Credential-Diebstahl, Privilege Escalation und laterale Bewegung zu einer kohärenten mehrstufigen Angriffskette kombiniert werden, statt der isolierten, werkzeuggetriebenen Schwachstellenaufzählung, die für einen automatisierten Scan typisch ist. Ihr Wert liegt genau in diesem Realismus: Sie testet, ob eine Kette einzeln geringfügiger Schwächen zu ernsthafter Kompromittierung kombiniert werden kann, und ob die Erkennungs- und Reaktionsfähigkeiten der Organisation einen Angriff dieser Form tatsächlich erfassen würden, während er sich entfaltet, nicht nur, ob ein einzelner Fehler isoliert existiert. Legacy-Systeme sind ein natürlicher Fokus für Red-Team-Übungen, weil ihre lange Betriebsgeschichte tendenziell genau die Art übersehener, sich verstärkender Schwächen angesammelt hat, die eine verkettete Attacke ausnutzt — ein ungepatchter Endpunkt, an den sich niemand erinnert und der immer noch exponiert ist, eine Admin-Schnittstelle, die immer noch mit Standardanmeldedaten läuft, eingerichtet, bevor irgendjemand im aktuellen Team dazukam. Eine gut durchgeführte Übung produziert Evidenz, nicht nur Befunde: eine demonstrierte Angriffskette, die von einer extern zugänglichen Legacy-Komponente in ein internes System reichte, ist eine weit überzeugendere Basis, um Behebungsfinanzierung von der Führungsebene zu sichern, als es eine Liste theoretischer Schwachstellen je sein könnte. Die Praxis ist teuer, um sie gut durchzuführen, da qualifizierte Praktiker teuer sind und eine Übung mit schlecht kontrolliertem Umfang echte betriebliche Störungen verursachen kann, und ihre Befunde sind nur wertvoll, wenn die Organisation die Kapazität hat, tatsächlich auf sie zu reagieren — sonst wird ein Red-Team-Bericht nur zu einem weiteren Eintrag in einem bereits überwältigenden Rückstand unadressierter Legacy-System-Probleme.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie klare Einsatzregeln, Umfang und Ziele, bevor Sie Red-Team-Übungen beginnen
- Stellen Sie ein Team mit vielfältigen offensiven Sicherheitsfähigkeiten zusammen oder heuern Sie es an, das Netzwerk-, Anwendungs- und Social-Engineering-Vektoren abdeckt
- Konzentrieren Sie anfängliche Übungen auf Legacy-System-Grenzen, wo Sicherheitskontrollen typischerweise am schwächsten sind
- Simulieren Sie reale Angriffsszenarien einschließlich Credential-Diebstahl, Privilege Escalation und lateraler Bewegung
- Dokumentieren Sie alle Befunde mit Reproduktionsschritten und Evidenz für das verteidigende Team
- Führen Sie Debriefing-Sitzungen mit sowohl Red- als auch Blue-Teams durch, um gelernte Lektionen zu teilen
- Planen Sie regelmäßige Red-Team-Übungen, um zu validieren, dass Fixes wirksam sind und neue Schwachstellen erfasst werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Offenbart real ausnutzbare Schwachstellen, die automatisierte Werkzeuge übersehen
- Testet die Wirksamkeit von Erkennungs- und Reaktionsfähigkeiten unter realistischen Bedingungen
- Liefert konkrete Evidenz, um Sicherheitsinvestitionen gegenüber dem Management zu rechtfertigen
- Identifiziert Lücken in Legacy-System-Verteidigungen, bevor tatsächliche Angreifer es tun

**Kosten und Risiken:**
- Qualifizierte Red-Team-Praktiker sind teuer anzuheuern oder zu halten
- Übungen können Störungen verursachen, wenn der Umfang nicht sorgfältig kontrolliert wird
- Befunde können für Teams, die bereits mit Legacy-Wartung kämpfen, überwältigend sein
- Ohne angemessene Nachverfolgung werden Red-Team-Befunde nur ein weiterer Rückstand unbehobener Probleme
- Kann Spannungen zwischen Sicherheits- und Entwicklungsteams erzeugen, wenn nicht diplomatisch gemanagt

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Einzelhandelsunternehmen beauftragte ein externes Red Team, seine Legacy-E-Commerce-Plattform zu testen, die zwölf Jahre in Produktion gewesen war. Das Red Team entdeckte, dass ein ungepatchter API-Endpunkt unauthentifizierten Zugang zur Kundenbestellhistorie erlaubte und dass eine Legacy-Admin-Schnittstelle immer noch Standardanmeldedaten nutzte. Diese Befunde, kombiniert mit einer demonstrierten Angriffskette, die von der Webanwendung zum internen Bestandssystem wanderte, überzeugten die Führungsebene, einen dedizierten Sicherheits-Behebungssprint zu finanzieren. Die Nachfolgeübung drei Monate später bestätigte, dass alle kritischen Befunde adressiert worden waren.
