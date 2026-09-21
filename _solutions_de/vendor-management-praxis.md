---
title: Vendor-Management-Praxis
description: Behandlung jedes externen Lieferanten als verwaltetes
  Risiko mit einem benannten Verantwortlichen, nachverfolgten
  Verpflichtungen, getesteten Ausstiegsoptionen und einer Beziehung,
  die Meinungsverschiedenheiten übersteht.
category:
- Dependencies
- Management
- Business
problems:
- dependency-on-supplier
- vendor-relationship-strain
- legal-disputes
- poor-contract-design
- technology-isolation
- vendor-lock-in
- technology-lock-in
- obsolete-technologies
- integration-difficulties
- vendor-dependency-entrapment
- core-modification-of-standard-software
- implementation-partner-dependency
- voided-vendor-support
layout: solution
lang: de
en_slug: vendor-management-practice
related_solutions:
- slug: application-portfolio-inventory
  similarity: 0.65
- slug: dependency-management-strategy
  similarity: 0.6
- slug: continuous-dependency-updates
  similarity: 0.6
- slug: knowledge-sharing-practices
  similarity: 0.6
- slug: system-decommissioning
  similarity: 0.6
- slug: technology-radar
  similarity: 0.55
---

## Description

Eine Vendor-Management-Praxis ist der Satz von Routinen, durch die eine Organisation externe Lieferanten — Softwareanbieter, Hosting-Anbieter, ausgelagerte Entwicklungspartner und lizenzierte Komponenten — davon abhält, zu unverwalteten Risiken zu werden. Sie hat vier Teile: Jemand ist für jede Lieferantenbeziehung verantwortlich, die Verpflichtungen auf beiden Seiten sind protokolliert und werden tatsächlich geprüft, die Kosten und Machbarkeit des Verlassens sind bekannt statt angenommen, und die Beziehung wird bewusst gepflegt statt nur während Eskalationen. Legacy-Systeme sammeln über Jahrzehnte Lieferantenabhängigkeiten an, und sie sammeln sie schlecht an: die Person, die den Vertrag ausgehandelt hat, ist gegangen, der Vertrag selbst ist schwer zu finden, niemand hat bewertet, was der Ersatz der Komponente kosten würde, und die Beziehung besteht ausschließlich aus einer jährlichen Rechnung und gelegentlichen wütenden E-Mails. Jedes davon ist einzeln handhabbar und zusammen ist es, wie Organisationen enden, unfähig sich zu bewegen.

## How to Apply ◆

> In einer Legacy-Landschaft sind die kritischen Lieferanten oft die, an die niemand denkt — eine 2009 eingebettete lizenzierte Bibliothek, ein Datenfeed ohne Vertrag, den irgendjemand finden kann —, sodass die Praxis mit einem Inventar beginnt statt mit Verhandlung.

- **Inventarisieren Sie jede externe Abhängigkeit**, die ein Problem verursachen würde, wenn sie aufhörte: kommerzielle Software, Bibliotheken mit restriktiven Lizenzen, Datenfeeds, gehostete Dienste und ausgelagerte Entwicklung. Protokollieren Sie für jede, was sie tut, was sie kostet, wer die Beziehung intern besitzt, wo der Vertrag ist, und wann er sich erneuert. Viele Organisationen können diese Fragen für die Mehrheit ihrer Lieferanten nicht beantworten, und dies herauszufinden ist selbst das erste Ergebnis.
- **Weisen Sie einen benannten internen Verantwortlichen zu** pro Lieferant — nicht eine Abteilung. Der Verantwortliche ist zuständig dafür, die vertraglichen Bedingungen zu kennen, zu verfolgen, ob Verpflichtungen erfüllt werden, und der Ansprechpartner zu sein. Nicht besessene Lieferantenbeziehungen sind die, die als Krisen zutage treten.
- **Klassifizieren Sie nach Kritikalität und Ersetzbarkeit.** Ein Lieferant, der sowohl kritisch als auch schwer zu ersetzen ist, rechtfertigt aktives Management: bekannten Ausstiegspfad, getestete Alternativen, Eskalationskontakte. Einer, der leicht ersetzbar ist, braucht fast nichts. Einheitliche Rigorosität auf alle Lieferanten anzuwenden garantiert, dass die wichtigen dieselbe unzureichende Aufmerksamkeit erhalten wie die trivialen.
- **Kennen Sie die Ausstiegskosten, bevor Sie sie brauchen.** Dokumentieren Sie für jeden kritischen Lieferanten, was Ersatz beinhalten würde, ungefähr was er kosten würde, und wie lange es dauern würde. Dies muss kein Migrationsplan sein; eine ehrliche einseitige Schätzung verwandelt die Verhandlungsposition, weil eine Partei, die ihre Alternativen nicht kennt, keine hat.
- Verhandeln Sie für **Daten- und Schnittstellenportabilität** statt nur für Preis: Export in einem dokumentierten Format, Source-Code-Escrow wo angemessen, eine definierte Kündigungsfrist, und das Recht, Abnahmetests durchzuführen. Diese Bedingungen kosten bei der Unterzeichnung wenig und sind später unerreichbar.
- **Verifizieren Sie Verpflichtungen, statt sie anzunehmen.** Service-Levels, Support-Reaktionszeiten und Sicherheitsverpflichtungen sollten planmäßig gegen tatsächliche Leistung geprüft werden. Unverifizierte vertragliche Versprechen werden häufig nicht eingehalten, und die Entdeckung geschieht üblicherweise während eines Vorfalls.
- **Verfolgen Sie End-of-Support-Termine zentral** und behandeln Sie sie als Planungsinputs ein Jahr im Voraus. Nicht unterstützte Komponenten zu betreiben ist eine Entscheidung, und sie sollte explizit getroffen werden, mit angegebenem Risiko, statt durch Unaufmerksamkeit zu geschehen.
- **Pflegen Sie die Beziehung außerhalb von Eskalationen.** Ein geplantes vierteljährliches Gespräch mit einem kritischen Lieferanten, wenn nichts falsch ist, produziert bessere Ergebnisse, wenn etwas ist. Beziehungen, die nur aus Beschwerden bestehen, degradieren zu positionellem Konflikt, und Streitigkeiten sind weit teurer als die Meetings, die sie verhindert hätten.
- **Eskalieren Sie vertraglich, bevor Sie rechtlich eskalieren.** Dokumentieren Sie Probleme schriftlich, während sie auftreten, rufen Sie den eigenen Abhilfeprozess des Vertrags an, und führen Sie eine Aufzeichnung. Die meisten Lieferantenstreitigkeiten, die Anwälte erreichen, tun dies, weil nichts dokumentiert wurde, während es noch behebbar war.

## Tradeoffs ⇄

> Lieferanten ordentlich zu verwalten kostet laufenden administrativen Aufwand und etwas Goodwill, im Austausch dafür, während einer Krise nicht zu entdecken, dass Sie keine Optionen haben.

**Vorteile:**

- Lieferantenausfälle, Preiserhöhungen und Einstellungen hören auf, Notfälle zu sein, weil die Alternativen und ihre Kosten bereits bekannt sind.
- Die Verhandlungsposition verbessert sich erheblich, da ein dokumentierter Ausstiegspfad der einzige echte Hebel ist, den ein Kunde hat.
- Lock-in wird identifiziert, während es noch günstig zu adressieren ist, statt an dem Punkt, an dem die Entscheidung eines Lieferanten zur Einschränkung der Organisation geworden ist.
- Streitigkeiten sind seltener und weniger schwerwiegend, weil Verpflichtungen verfolgt und Probleme dokumentiert werden, während sie noch routinemäßig sind.
- End-of-Support-Überraschungen verschwinden weitgehend, was eine der häufigeren Ursachen ungeplanter, dringender Modernisierungsarbeit beseitigt.

**Kosten und Risiken:**

- Das Inventar und die laufende Verfolgung sind echte administrative Arbeit ohne sichtbare Ausgabe, und sie sind leicht zu deprioritisieren, bis ein Vorfall sie rückwirkend offensichtlich macht.
- Ausstiegsbewertungen kosten Aufwand für Optionen, die üblicherweise nie ausgeübt werden, und es ist schwierig zu rechtfertigen, sie aktuell zu halten.
- Portabilitätsanforderungen und Escrow-Bedingungen erhöhen den Preis eines Vertrags, manchmal erheblich, und der Nutzen ist bedingt.
- Die Formalisierung einer Beziehung kann als Misstrauen gelesen werden, besonders bei kleinen Lieferanten oder langjährigen Partnern, wo die informelle Beziehung gut funktioniert hat.
- Die Pflege einer getesteten Alternative zu einem kritischen Lieferanten ist genuin teuer, und für viele Abhängigkeiten ist die ehrliche Antwort, das Risiko zu akzeptieren und diese Akzeptanz zu dokumentieren, statt es zu mindern.

## How It Could Be

Ein Finanzdienstleistungsunternehmen verließ sich seit 2011 auf eine Drittanbieter-Regel-Engine, die in sein Kernsystem eingebettet war. Der Anbieter kündigte End of Support mit zwölf Monaten Vorlaufzeit an. Niemand intern besaß die Beziehung, der Vertrag brauchte zwei Wochen zum Auffinden, und es stellte sich heraus, dass kein Exportformat für die angesammelten Regeldefinitionen vertraglich garantiert war. Die Extraktion von ungefähr 4.000 Regeln aus einem proprietären Binärformat verbrauchte neun Monate und zwei Vollzeitentwickler. Im Nachgang inventarisierte das Unternehmen seine rund 60 externen Abhängigkeiten, identifizierte neun als kritisch und schwer zu ersetzen, und produzierte für jede eine einseitige Ausstiegsbewertung. Die erste Bewertung, die sie schrieben — für einen Dokumentgenerierungsdienst — offenbarte, dass eine Abwanderung drei Monate dauern würde, was sie im folgenden Jahr nutzten, um eine Verlängerung 30 Prozent unter der Eröffnungsposition des Anbieters auszuhandeln.

Eine andere Organisation vermied eine Streitigkeit durch Dokumentation statt Eskalation. Ihr ausgelagerter Wartungspartner hatte das vertragliche Vier-Stunden-Reaktionsziel für kritische Vorfälle verfehlt, aber die Fehlschläge wurden einzeln bemerkt und vergessen. Nach der Zuweisung eines Verantwortlichen, der Reaktionszeiten gegen die Vereinbarung verfolgte, wurde das Muster sichtbar: 40 Prozent der kritischen Vorfälle über sechs Monate hatten das Ziel überschritten. Präsentiert in einem geplanten vierteljährlichen Review als Daten statt als Vorwurf, führte es dazu, dass der Lieferant seine Bereitschaftsabdeckung für das Konto umstrukturierte. Der alternative Pfad, den das Rechtsteam der Organisation zu vorbereiten begonnen hatte, wäre ein vertraglicher Streit über eine Ansammlung individuell abstreitbarer Vorfälle gewesen.
